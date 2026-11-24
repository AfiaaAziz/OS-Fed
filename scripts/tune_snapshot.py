"""Step 8.10: tune the unstated OS-Fed snapshot settings (decisions D4, D5).

One client, one round. We train one client locally once, then try a few
snapshot initializations and L-BFGS iteration counts, and look ONLY at:
  * match_loss  (Eq. 5, should be SMALL)
  * cos_sim     (cosine similarity between the true update Δw and the
                 synthetic gradient, should be clearly POSITIVE)
  * residual_ratio (||e_new|| / ||Δg||, should not be close to or above 1)
We never look at test accuracy here (plan rule 21). Error compensation is off
(e = 0) because this is a single round.

Run from the project root, e.g.:
    python -m scripts.tune_snapshot --config configs/A_osfed_cnn_noniid_g1ec0_smoke_s1.yaml
    python -m scripts.tune_snapshot --config <cfg> --client-id 0 \
        --inits first_client_image random --iters 10 20 50
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from src.client import Client, clone_state, train_local
from src.data import FastLoader, load_cifar10
from src.model import build_model
from src.osfed import local_delta, make_initial_image, optimize_snapshot
from src.partition import load_partition
from src.utils import get_device, project_root, set_seed


def build_one_client(cfg: dict, client_id: int, device: torch.device) -> Client:
    root = project_root()
    data = load_cifar10()
    parts, _meta = load_partition(
        root / "data" / "partitions" / f"{cfg['partition']}.json"
    )
    indices = parts[client_id]

    samples = cfg.get("samples_per_client")
    if samples:
        rng = np.random.default_rng(int(cfg["seed"]) * 1000 + client_id)
        count = min(int(samples), len(indices))
        indices = np.sort(rng.choice(indices, size=count, replace=False))

    loader = FastLoader(
        data.x_train,
        data.y_train,
        indices=indices,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        seed=int(cfg["seed"]) * 100000 + client_id * 1000,
        device=device,
    )
    return Client(client_id, loader)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="an OS-Fed yaml config")
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--inits", nargs="+", default=["first_client_image", "random"])
    ap.add_argument("--iters", nargs="+", type=int, default=[10, 20, 50])
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    seed = int(cfg["seed"])
    grid_size = int(cfg["grid_size"])
    num_classes = int(cfg.get("num_classes", 10))
    shape = tuple(int(v) for v in cfg.get("snapshot_shape", [3, 32, 32]))
    eps = float(cfg["snapshot_eps"])
    lbfgs_lr = float(cfg["lbfgs_lr"])

    print(f"device       : {device}")
    print(f"config       : {args.config}")
    print(f"model/grid   : {cfg['model']}  grid={grid_size}x{grid_size}")
    print(f"client id    : {args.client_id}")
    print(f"inits        : {args.inits}")
    print(f"L-BFGS iters : {args.iters}")

    # --- one client, one round: build the initial global model, train once ---
    set_seed(seed)
    model = build_model(cfg["model"], num_classes, cfg.get("norm", "batch")).to(device)
    global_state = clone_state(model)

    client = build_one_client(cfg, args.client_id, device)
    print(f"client samples: {client.n_samples}")

    local_state, train_loss, train_s = train_local(
        model,
        global_state,
        client.loader,
        epochs=int(cfg["local_epochs"]),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
        device=device,
    )
    delta_w = local_delta(model, global_state, local_state)
    target = delta_w  # e = 0 for a single tuning round
    print(
        f"local train  : loss={train_loss:.4f} time={train_s:.1f}s  "
        f"||dW||={target.norm().item():.4f}  dims={target.numel()}"
    )

    # The matching model stays frozen at the global weights, in eval mode.
    model.load_state_dict(global_state)
    model.to(device)
    model.eval()
    dtype = next(model.parameters()).dtype

    rows = []
    print()
    header = (
        f"{'init':<20}{'iters':>6}{'match_loss':>14}"
        f"{'cos_sim':>10}{'resid_ratio':>13}{'opt_s':>8}"
    )
    print(header)
    print("-" * len(header))

    for init in args.inits:
        for max_iter in args.iters:
            gen = torch.Generator()
            gen.manual_seed(seed * 100003 + max_iter)

            image = nn.Parameter(
                make_initial_image(init, client, shape, device, dtype, generator=gen)
                .detach().clone()
            )
            label_logits = nn.Parameter(
                torch.zeros(grid_size * grid_size, num_classes, device=device, dtype=dtype)
            )

            t0 = time.perf_counter()
            res = optimize_snapshot(
                model=model,
                image=image,
                label_logits=label_logits,
                target_delta=target,
                grid_size=grid_size,
                max_iter=max_iter,
                lbfgs_lr=lbfgs_lr,
                eps=eps,
            )
            opt_s = time.perf_counter() - t0

            rows.append({
                "init": init,
                "iters": max_iter,
                "match_loss": res["match_loss"],
                "cos_sim": res["cosine_similarity"],
                "residual_ratio": res["residual_ratio"],
                "opt_s": round(opt_s, 3),
            })
            print(
                f"{init:<20}{max_iter:>6}{res['match_loss']:>14.6f}"
                f"{res['cosine_similarity']:>10.4f}"
                f"{res['residual_ratio']:>13.4f}{opt_s:>8.2f}"
            )

    out = project_root() / "results" / "E0_sanity" / (
        f"tune_snapshot_{cfg['model']}_g{grid_size}_client{args.client_id}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["init", "iters", "match_loss", "cos_sim", "residual_ratio", "opt_s"],
        )
        w.writeheader()
        w.writerows(rows)

    print("\nsaved:", out)
    print("\nHow to read this (plan Step 8.10):")
    print("  * Pick the setting where match_loss is clearly LOW and cos_sim clearly POSITIVE,")
    print("    using the FEWEST L-BFGS iters that already gets there (cheapest good option).")
    print("  * Then paste this table back and we record D4 (init) and D5 (iters) in")
    print("    notes/decisions.md before any full OS-Fed run.")


if __name__ == "__main__":
    main()
