"""Step 9 check: compare a multi-process run with the single-process run of the SAME config.

    python -m scripts.compare_runs --single logs/<single_run_id> --proc logs/<proc_run_id> [--acc-tol 0.01]

Checks: same number of rounds, identical byte counts, test_acc close, and that the process run
really logged per-client compute time and server wait time. Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True)
    ap.add_argument("--proc", required=True)
    ap.add_argument("--acc-tol", type=float, default=0.01, help="largest allowed |test_acc difference| per round")
    args = ap.parse_args()

    a = pd.read_csv(Path(args.single) / "rounds.csv")
    b = pd.read_csv(Path(args.proc) / "rounds.csv")
    ca_path, cb_path = Path(args.single) / "clients.csv", Path(args.proc) / "clients.csv"
    ca = pd.read_csv(ca_path) if ca_path.exists() else None
    cb = pd.read_csv(cb_path) if cb_path.exists() else None

    print(f"single: {args.single}\nproc  : {args.proc}\n")
    n = min(len(a), len(b))
    cols = ["round", "test_acc", "upload_bytes_total", "download_bytes_total"]
    table = pd.DataFrame({
        "round": a["round"][:n],
        "acc_single": a["test_acc"][:n].values, "acc_proc": b["test_acc"][:n].values,
        "acc_diff": (a["test_acc"][:n].values - b["test_acc"][:n].values),
    })
    print(table.to_string(index=False), "\n")

    checks = []
    checks.append(("same number of rounds", len(a) == len(b), f"{len(a)} vs {len(b)}"))
    checks.append(("upload bytes identical every round",
                   bool((a["upload_bytes_total"][:n].values == b["upload_bytes_total"][:n].values).all()), ""))
    checks.append(("download bytes identical every round",
                   bool((a["download_bytes_total"][:n].values == b["download_bytes_total"][:n].values).all()), ""))
    max_diff = float(table["acc_diff"].abs().max())
    checks.append((f"test_acc differs by at most {args.acc_tol}", max_diff <= args.acc_tol, f"max diff {max_diff:.4f}"))

    has_wait = "server_wait_s" in b.columns
    checks.append(("proc log has server_wait_s > 0 in some round",
                   has_wait and bool((b["server_wait_s"] > 0).any()),
                   "all zero: the server wait time is not being measured" if has_wait else "column missing"))
    checks.append(("proc log has max_client_compute_s > 0",
                   "max_client_compute_s" in b.columns and bool((b["max_client_compute_s"] > 0).all()), ""))
    if ca is not None and cb is not None:
        per_a = ca.groupby("round").size().tolist()
        per_b = cb.groupby("round").size().tolist()
        checks.append(("clients.csv: same number of client rows per round", per_a == per_b, f"{per_a} vs {per_b}"))
        checks.append(("clients.csv: local_train_s > 0 for every client", bool((cb["local_train_s"] > 0).all()), ""))
    else:
        checks.append(("clients.csv exists in both runs", False, "missing"))

    ok_all = True
    for name, ok, detail in checks:
        ok_all &= bool(ok)
        shown = detail if (not ok or name.startswith(("same number", "test_acc"))) else ""
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {shown}")
    print("\nOVERALL:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
