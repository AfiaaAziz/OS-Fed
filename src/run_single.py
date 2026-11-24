"""Run one federated experiment in a single process."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
    payload_bytes,
    wire_bytes,
)
from src.model import build_model
from src.osfed import OSFed
from src.partition import load_partition
from src.server import Server
from src.utils import env_report, get_device, project_root, set_seed, today_stamp


METHODS = {
    "fedavg": FedAvg,
    "osfed": OSFed,
}

REQUIRED = [
    "tier",
    "method",
    "model",
    "partition",
    "n_clients",
    "rounds",
    "local_epochs",
    "batch_size",
    "lr",
    "lr_schedule",
    "momentum",
    "weight_decay",
    "seed",
    "aggregation",
]


class Tee:
    """Write printed output to both the terminal and a log file."""

    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def save_checkpoint(
    ckpt_dir: Path,
    round_idx: int,
    model_state,
    cum_wall_s: float,
    cfg: dict,
    run_id: str,
    keep_round_file: bool,
    method_state: dict | None = None,
) -> None:
    """Save the global model and method-specific state."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    blob = {
        "round": round_idx,
        "model_state": model_state,
        "method_state": method_state or {},
        "cum_wall_s": cum_wall_s,
        "run_id": run_id,
        "config": cfg,
    }

    temporary_path = ckpt_dir / "latest.tmp"
    torch.save(blob, temporary_path)
    os.replace(temporary_path, ckpt_dir / "latest.pt")

    if keep_round_file:
        torch.save(blob, ckpt_dir / f"round_{round_idx:03d}.pt")


def append_run_log(
    root: Path,
    run_id: str,
    cfg: dict,
    status: str,
    notes: str = "",
) -> None:
    """Append a run status to notes/run_log.md."""
    path = root / "notes" / "run_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    who = os.environ.get("USER", "")
    line = (
        f"| {run_id} | {datetime.now():%Y-%m-%d} | {who} | "
        f"{cfg.get('purpose', '')} | {status} | {notes} |\n"
    )

    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as file:
        if new_file:
            file.write(
                "| run_id | date | who | purpose | status | notes |\n"
                "|---|---|---|---|---|---|\n"
            )
        file.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run ID to continue from checkpoints/<run_id>/latest.pt",
    )
    parser.add_argument("--condition", type=str, default="net0")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Stop after this round number; resume later",
    )
    args = parser.parse_args()

    root = project_root()

    if args.resume:
        run_id = args.resume
        run_dir = root / "logs" / run_id
        config_path = run_dir / "config_used.yaml"
        if not config_path.exists():
            raise SystemExit(f"Cannot find saved config: {config_path}")
        with open(config_path, encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    else:
        if not args.config:
            parser.error("Provide --config or --resume")

        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"Cannot find config: {config_path}")
        with open(config_path, encoding="utf-8") as file:
            cfg = yaml.safe_load(file)

        run_id = f"{today_stamp()}_{config_path.stem}__{args.condition}"
        run_dir = root / "logs" / run_id

    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise SystemExit(f"Config is missing required keys: {missing}")

    if cfg["method"] not in METHODS:
        raise SystemExit(
            f"Unknown method {cfg['method']!r}; "
            f"choose one of {sorted(METHODS)}"
        )

    if args.resume:
        latest = root / "checkpoints" / run_id / "latest.pt"
        if not latest.exists():
            raise SystemExit(f"Cannot find checkpoint: {latest}")

    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        if (run_dir / "rounds.csv").exists():
            raise SystemExit(
                f"{run_dir} already has results. "
                "Use --resume or change --condition."
            )
        with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(cfg, file, sort_keys=False)

    tee = Tee(run_dir / "stdout.txt")
    sys.stdout = tee
    try:
        run_experiment(cfg, run_id, run_dir, root, args)
    finally:
        sys.stdout = tee.stdout
        tee.close()


def run_experiment(cfg: dict, run_id: str, run_dir: Path, root: Path, args) -> None:
    device = get_device()
    method_name = cfg["method"]
    n_clients = int(cfg["n_clients"])
    total_rounds = int(cfg["rounds"])
    local_epochs = int(cfg["local_epochs"])
    seed = int(cfg["seed"])

    ckpt_dir = root / "checkpoints" / run_id
    checkpoint_every = int(cfg.get("ckpt_every", 1))

    with open(run_dir / "env.txt", "a", encoding="utf-8") as file:
        file.write(env_report())
        file.write(f"run_id: {run_id}\ndevice: {device}\n\n")

    print(f"run_id : {run_id}")
    print(f"device : {device}")
    print(f"method : {method_name}")
    print(f"purpose: {cfg.get('purpose', '')}")

    data = load_cifar10()
    partition_path = root / "data" / "partitions" / f"{cfg['partition']}.json"
    parts, _partition_metadata = load_partition(partition_path)

    if n_clients > len(parts):
        raise SystemExit(
            f"Partition has {len(parts)} clients, config asks for {n_clients}"
        )

    set_seed(seed)
    model_name = cfg["model"]
    norm = cfg.get("norm", "batch")

    server = Server(
        build_model(model_name, 10, norm),
        FastLoader(
            data.x_test,
            data.y_test,
            batch_size=500,
            shuffle=False,
            device=device,
        ),
        device,
    )
    work_model = build_model(model_name, 10, norm)
    method = METHODS[method_name](cfg, work_model, device)

    start_round = 0
    cumulative_wall_s = 0.0
    latest_checkpoint = ckpt_dir / "latest.pt"

    if args.resume:
        checkpoint = torch.load(
            latest_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        server.set_state(checkpoint["model_state"])
        start_round = int(checkpoint["round"])
        cumulative_wall_s = float(checkpoint["cum_wall_s"])

        if hasattr(method, "load_state_dict"):
            method.load_state_dict(checkpoint.get("method_state", {}))

        print(f"resumed from round {start_round}")

    samples_per_client = cfg.get("samples_per_client")
    clients = []

    for client_id in range(n_clients):
        indices = parts[client_id]

        if samples_per_client:
            rng = np.random.default_rng(seed * 1000 + client_id)
            count = min(int(samples_per_client), len(indices))
            indices = np.sort(
                rng.choice(indices, size=count, replace=False)
            )

        loader = FastLoader(
            data.x_train,
            data.y_train,
            indices=indices,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            seed=seed * 100000 + client_id * 1000,
            device=device,
        )
        loader.set_epoch(start_round * local_epochs)
        clients.append(Client(client_id, loader))

    print("client sizes:", [client.n_samples for client in clients])

    if method_name == "osfed":
        rounds_columns = ROUNDS_COLUMNS_OSFED
        clients_columns = CLIENTS_COLUMNS_OSFED
    else:
        rounds_columns = ROUNDS_COLUMNS
        clients_columns = CLIENTS_COLUMNS

    rounds_log = CsvLogger(run_dir / "rounds.csv", rounds_columns)
    clients_log = CsvLogger(run_dir / "clients.csv", clients_columns)

    if not args.resume:
        append_run_log(root, run_id, cfg, "started")

    initial_model_bytes = payload_bytes(server.get_state())
    first_upload_total = None
    recent_accuracies = []
    last_round = start_round

    for round_number in range(start_round + 1, total_rounds + 1):
        round_start = time.perf_counter()

        if cfg["lr_schedule"] == "cosine":
            learning_rate = cosine_lr(
                float(cfg["lr"]),
                round_number - 1,
                total_rounds,
            )
        else:
            learning_rate = float(cfg["lr"])

        global_state = server.get_state()
        download_bytes_per_client = payload_bytes(global_state)

        messages = []
        counts = []
        losses = []
        client_times = []
        client_infos = []
        upload_total = 0

        for client in clients:
            message, info = method.client_update(
                client,
                global_state,
                learning_rate,
            )

            upload_bytes = payload_bytes(message)
            upload_total += upload_bytes
            messages.append(message)
            counts.append(client.n_samples)
            losses.append(float(info["train_loss"]))
            client_times.append(
                float(info["local_train_s"]) + float(info["snapshot_opt_s"])
            )
            client_infos.append(info)

            client_row = {
                "run_id": run_id,
                "round": round_number,
                "client_id": client.client_id,
                "n_samples": client.n_samples,
                "local_train_s": round(float(info["local_train_s"]), 3),
                "snapshot_opt_s": round(float(info["snapshot_opt_s"]), 3),
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes_per_client,
            }

            if method_name == "osfed":
                client_row.update(
                    match_loss_final=float(info["match_loss_final"]),
                    cos_sim=float(info["cos_sim"]),
                    residual_ratio=float(info["residual_ratio"]),
                )

            clients_log.log(**client_row)

        if (
            round_number == start_round + 1
            and not (run_dir / "bytes_summary.json").exists()
        ):
            summary = {
                "download_payload_bytes_per_client": download_bytes_per_client,
                "download_wire_bytes_per_client": wire_bytes(global_state),
                "upload_payload_bytes_per_client": payload_bytes(messages[0]),
                "upload_wire_bytes_per_client": wire_bytes(messages[0]),
            }
            (run_dir / "bytes_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

        method.server_update(server, messages, counts)
        test_loss, test_accuracy = server.evaluate()

        round_seconds = time.perf_counter() - round_start
        cumulative_wall_s += round_seconds
        download_total = download_bytes_per_client * len(clients)

        if first_upload_total is None:
            first_upload_total = upload_total

        round_row = {
            "run_id": run_id,
            "round": round_number,
            "test_acc": round(test_accuracy, 4),
            "test_loss": round(test_loss, 4),
            "upload_bytes_total": upload_total,
            "download_bytes_total": download_total,
            "round_wall_s": round(round_seconds, 3),
            "server_wait_s": 0.0,
            "cum_wall_s": round(cumulative_wall_s, 3),
            "max_client_compute_s": round(max(client_times), 3),
        }

        if method_name == "osfed":
            round_row.update(
                match_loss_mean=float(
                    np.mean(
                        [info["match_loss_final"] for info in client_infos]
                    )
                ),
                cos_sim_mean=float(
                    np.mean([info["cos_sim"] for info in client_infos])
                ),
                residual_ratio_mean=float(
                    np.mean(
                        [info["residual_ratio"] for info in client_infos]
                    )
                ),
            )

        rounds_log.log(**round_row)
        recent_accuracies.append(test_accuracy)

        print(
            f"[round {round_number:3d}/{total_rounds}] "
            f"lr={learning_rate:.5f} "
            f"train_loss={np.mean(losses):.4f} "
            f"test_acc={test_accuracy:.4f} "
            f"test_loss={test_loss:.4f} "
            f"up={upload_total / 1e6:.3f}MB "
            f"down={download_total / 1e6:.3f}MB "
            f"round={round_seconds:.1f}s "
            f"total={cumulative_wall_s / 60:.1f}min"
        )

        method_state = (
            method.state_dict()
            if hasattr(method, "state_dict")
            else {}
        )

        save_checkpoint(
            ckpt_dir=ckpt_dir,
            round_idx=round_number,
            model_state=server.get_state(),
            cum_wall_s=cumulative_wall_s,
            cfg=cfg,
            run_id=run_id,
            keep_round_file=(
                round_number % checkpoint_every == 0
                or round_number == total_rounds
            ),
            method_state=method_state,
        )

        last_round = round_number

        if args.stop_after and round_number >= args.stop_after:
            print(
                f"Stopped after round {round_number}. Resume with:\n"
                f"python -m src.run_single --resume {run_id}"
            )
            break

    rounds_log.close()
    clients_log.close()

    print("\n=== checks ===")
    if first_upload_total is not None:
        if method_name == "fedavg":
            expected_upload_total = initial_model_bytes * n_clients
            matches = first_upload_total == expected_upload_total
            print(
                f"FedAvg upload bytes: {first_upload_total:,}; "
                f"expected {expected_upload_total:,} -> "
                f"{'OK' if matches else 'MISMATCH'}"
            )
        else:
            print(
                "OS-Fed snapshot upload bytes in first logged round: "
                f"{first_upload_total:,} total"
            )

    if len(recent_accuracies) > 1:
        print(
            f"accuracy: round {last_round - len(recent_accuracies) + 1} "
            f"= {recent_accuracies[0]:.4f} -> round {last_round} "
            f"= {recent_accuracies[-1]:.4f}"
        )

    if last_round >= total_rounds:
        final_note = (
            f"final test_acc={recent_accuracies[-1]:.4f}"
            if recent_accuracies
            else ""
        )
        append_run_log(root, run_id, cfg, "done", final_note)

    print("logs       :", run_dir)
    print("checkpoints:", ckpt_dir)


if __name__ == "__main__":
    main()