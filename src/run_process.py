"""Step 9 (Phase 4): run one federated experiment with SEPARATE client processes.

One server process (this one) + N client worker processes. The server broadcasts
the global model each round, every client trains in its own process, the server
waits for all of them, aggregates, evaluates and logs. This reuses the SAME
tested building blocks as run_single.py (client.py, server.py, fedavg.py,
osfed.py, metrics.py); only the process orchestration is new here.

Plan notes honoured:
  * spawn start method; each worker limited to 1 PyTorch thread (4-core VM).
  * messages crossing the process boundary are serialized to bytes, so the byte
    count is exact (we log payload bytes to stay comparable with run_single.py,
    and also record the serialized wire size in bytes_summary.json).
  * the server waits for ALL clients each round, with a long safety timeout that
    is logged if it ever triggers.
  * per-client compute time and the server's waiting time are logged separately.
  * OS-Fed error state lives inside each client process.

NOTE: this file is written on Windows but is meant to run on the Ubuntu VM
(small CNN, CPU). It is syntax-checked, not executed here. The plan's check is:
a 3-round process run with the same seed should match the single-process run
(tiny numeric differences allowed). FedAvg matches closely; OS-Fed uses one RNG
per process so it will be close, not bit-identical. Test on the VM.

Run from the project root:
    python -m src.run_process --config configs/A_fedavg_cnn_noniid_r3_s1.yaml
    python -m src.run_process --config <cfg> --clients 2        # start small
"""
from __future__ import annotations

import argparse
import json
import queue as queue_mod
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from src.client import Client
from src.data import FastLoader, load_cifar10
from src.fedavg import FedAvg, cosine_lr
from src.metrics import (
    CLIENTS_COLUMNS,
    CLIENTS_COLUMNS_OSFED,
    ROUNDS_COLUMNS,
    ROUNDS_COLUMNS_OSFED,
    CsvLogger,
    from_bytes,
    payload_bytes,
    to_bytes,
    wire_bytes,
)
from src.model import build_model
from src.osfed import OSFed
from src.partition import load_partition
from src.server import Server
from src.utils import env_report, get_device, project_root, set_seed, today_stamp

METHODS = {"fedavg": FedAvg, "osfed": OSFed}

REQUIRED = [
    "tier", "method", "model", "partition", "n_clients", "rounds",
    "local_epochs", "batch_size", "lr", "lr_schedule", "momentum",
    "weight_decay", "seed", "aggregation",
]


# --------------------------------------------------------------------------- #
# Client worker process                                                       #
# --------------------------------------------------------------------------- #
def client_worker(client_id, cfg, indices, device_str, in_q, out_q):
    """One client lives here for the whole run. Reusable across rounds."""
    torch.set_num_threads(1)  # 10 processes must not fight over 4 cores
    device = torch.device(device_str)
    seed = int(cfg["seed"])
    set_seed(seed + client_id)

    # Build this client's data loader once (same recipe as run_single.py).
    data = load_cifar10()
    loader = FastLoader(
        data.x_train, data.y_train,
        indices=np.asarray(indices),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        seed=seed * 100000 + client_id * 1000,
        device=device,
    )
    del data  # keep only this client's subset in memory
    loader.set_epoch(int(cfg.get("start_epoch", 0)))
    client = Client(client_id, loader)

    # Each client owns its own method instance (OS-Fed error state is per-client).
    work_model = build_model(cfg["model"], 10, cfg.get("norm", "batch"))
    method = METHODS[cfg["method"]](cfg, work_model, device)

    out_q.put(("READY", client_id))

    while True:
        cmd, payload = in_q.get()
        if cmd == "STOP":
            break
        if cmd != "ROUND":
            continue

        global_state_bytes, lr = payload
        global_state = from_bytes(global_state_bytes)

        t0 = time.perf_counter()
        message, info = method.client_update(client, global_state, float(lr))
        compute_s = time.perf_counter() - t0

        message_bytes = to_bytes(message)
        out_q.put((
            "RESULT",
            {
                "client_id": client_id,
                "n_samples": client.n_samples,
                "message_bytes": message_bytes,            # serialized (exact wire)
                "payload_bytes": payload_bytes(message),   # tensor bytes (comparable)
                "wire_bytes": len(message_bytes),
                "info": info,
                "compute_s": compute_s,
            },
        ))


# --------------------------------------------------------------------------- #
# Server (main process)                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", default="net0")
    parser.add_argument("--clients", type=int, default=0,
                        help="override number of clients (for the 2-client first test)")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="server safety timeout per client response, seconds")
    args = parser.parse_args()

    root = project_root()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"Config is missing required keys: {missing}")
    if cfg["method"] not in METHODS:
        raise SystemExit(f"Unknown method {cfg['method']!r}")

    n_clients = args.clients or int(cfg["n_clients"])
    total_rounds = int(cfg["rounds"])
    local_epochs = int(cfg["local_epochs"])
    seed = int(cfg["seed"])
    method_name = cfg["method"]

    config_stem = Path(args.config).stem
    run_id = f"{today_stamp()}_{config_stem}__{args.condition}_proc"
    run_dir = root / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "rounds.csv").exists():
        raise SystemExit(f"{run_dir} already has results. Change --condition.")
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "env.txt", "a", encoding="utf-8") as f:
        f.write(env_report())
        f.write(f"run_id: {run_id}\nmode: multiprocess\nn_clients: {n_clients}\n\n")

    device = get_device()
    print(f"run_id : {run_id}")
    print(f"device : {device}  (processes: {n_clients} clients + 1 server)")
    print(f"method : {method_name}")

    # Partition and server model.
    parts, _meta = load_partition(
        root / "data" / "partitions" / f"{cfg['partition']}.json"
    )
    if n_clients > len(parts):
        raise SystemExit(f"Partition has {len(parts)} clients, asked for {n_clients}")

    data = load_cifar10()
    set_seed(seed)
    server = Server(
        build_model(cfg["model"], 10, cfg.get("norm", "batch")),
        FastLoader(data.x_test, data.y_test, batch_size=500, shuffle=False, device=device),
        device,
    )
    # Server-side method instance (used only for server_update / aggregation).
    server_method = METHODS[method_name](cfg, build_model(cfg["model"], 10, cfg.get("norm", "batch")), device)
    del data

    # Loggers (same columns as run_single.py).
    if method_name == "osfed":
        rounds_columns, clients_columns = ROUNDS_COLUMNS_OSFED, CLIENTS_COLUMNS_OSFED
    else:
        rounds_columns, clients_columns = ROUNDS_COLUMNS, CLIENTS_COLUMNS
    rounds_log = CsvLogger(run_dir / "rounds.csv", rounds_columns)
    clients_log = CsvLogger(run_dir / "clients.csv", clients_columns)

    # --- start the client processes (spawn) ---
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    in_qs = {}
    procs = {}
    for cid in range(n_clients):
        in_q = ctx.Queue()
        in_qs[cid] = in_q
        p = ctx.Process(
            target=client_worker,
            args=(cid, cfg, parts[cid].tolist(), str(device), in_q, out_q),
            daemon=True,
        )
        p.start()
        procs[cid] = p

    # wait for every worker to report READY
    ready = 0
    while ready < n_clients:
        tag, _ = out_q.get(timeout=args.timeout)
        if tag == "READY":
            ready += 1
    print(f"all {n_clients} client processes ready")

    cumulative_wall_s = 0.0
    timeouts = 0

    for round_number in range(1, total_rounds + 1):
        round_start = time.perf_counter()

        lr = (cosine_lr(float(cfg["lr"]), round_number - 1, total_rounds)
              if cfg["lr_schedule"] == "cosine" else float(cfg["lr"]))

        global_state = server.get_state()
        global_state_bytes = to_bytes(global_state)
        download_bytes_per_client = payload_bytes(global_state)

        # broadcast the model to every client
        for cid in range(n_clients):
            in_qs[cid].put(("ROUND", (global_state_bytes, lr)))

        # collect all responses; measure how long we wait after the FIRST one
        results = {}
        first_arrival = None
        last_arrival = None
        while len(results) < n_clients:
            try:
                tag, payload = out_q.get(timeout=args.timeout)
            except queue_mod.Empty:
                timeouts += 1
                print(f"!! timeout waiting for clients in round {round_number} "
                      f"(got {len(results)}/{n_clients})")
                raise
            if tag != "RESULT":
                continue
            now = time.perf_counter()
            if first_arrival is None:
                first_arrival = now
            last_arrival = now
            results[payload["client_id"]] = payload

        server_wait_s = (last_arrival - first_arrival) if first_arrival else 0.0

        # rebuild messages in client-id order (so aggregation matches run_single)
        ordered = [results[cid] for cid in range(n_clients)]
        messages = [from_bytes(r["message_bytes"]) for r in ordered]
        counts = [r["n_samples"] for r in ordered]

        upload_total = 0
        client_compute_times = []
        client_infos = []
        for r in ordered:
            info = r["info"]
            client_infos.append(info)
            upload_total += r["payload_bytes"]
            client_compute_times.append(
                float(info["local_train_s"]) + float(info["snapshot_opt_s"])
            )
            row = {
                "run_id": run_id, "round": round_number, "client_id": r["client_id"],
                "n_samples": r["n_samples"],
                "local_train_s": round(float(info["local_train_s"]), 3),
                "snapshot_opt_s": round(float(info["snapshot_opt_s"]), 3),
                "upload_bytes": r["payload_bytes"],
                "download_bytes": download_bytes_per_client,
            }
            if method_name == "osfed":
                row.update(match_loss_final=float(info["match_loss_final"]),
                           cos_sim=float(info["cos_sim"]),
                           residual_ratio=float(info["residual_ratio"]))
            clients_log.log(**row)

        # one-time byte summary (payload + serialized wire size)
        if round_number == 1 and not (run_dir / "bytes_summary.json").exists():
            (run_dir / "bytes_summary.json").write_text(json.dumps({
                "download_payload_bytes_per_client": download_bytes_per_client,
                "download_wire_bytes_per_client": wire_bytes(global_state),
                "upload_payload_bytes_per_client": ordered[0]["payload_bytes"],
                "upload_wire_bytes_per_client": ordered[0]["wire_bytes"],
            }, indent=2), encoding="utf-8")

        # aggregate + evaluate on the server
        server_method.server_update(server, messages, counts)
        test_loss, test_acc = server.evaluate()

        round_seconds = time.perf_counter() - round_start
        cumulative_wall_s += round_seconds
        download_total = download_bytes_per_client * n_clients

        row = {
            "run_id": run_id, "round": round_number,
            "test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4),
            "upload_bytes_total": upload_total, "download_bytes_total": download_total,
            "round_wall_s": round(round_seconds, 3),
            "server_wait_s": round(server_wait_s, 3),
            "cum_wall_s": round(cumulative_wall_s, 3),
            "max_client_compute_s": round(max(client_compute_times), 3),
        }
        if method_name == "osfed":
            row.update(
                match_loss_mean=float(np.mean([i["match_loss_final"] for i in client_infos])),
                cos_sim_mean=float(np.mean([i["cos_sim"] for i in client_infos])),
                residual_ratio_mean=float(np.mean([i["residual_ratio"] for i in client_infos])),
            )
        rounds_log.log(**row)

        print(f"[round {round_number:3d}/{total_rounds}] "
              f"test_acc={test_acc:.4f} up={upload_total/1e6:.3f}MB "
              f"down={download_total/1e6:.3f}MB "
              f"server_wait={server_wait_s:.2f}s round={round_seconds:.1f}s")

    # shut the workers down
    for cid in range(n_clients):
        in_qs[cid].put(("STOP", None))
    for p in procs.values():
        p.join(timeout=10)

    rounds_log.close()
    clients_log.close()
    print(f"\ndone. timeouts triggered: {timeouts}")
    print("logs:", run_dir)


if __name__ == "__main__":
    main()
