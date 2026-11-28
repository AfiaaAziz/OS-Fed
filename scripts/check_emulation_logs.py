"""Step 10 check on a finished emulated run (run_process with --condition ...).

    python -m scripts.check_emulation_logs --run logs/<run_id>

The condition is read from the run folder name (the part after '__', without a trailing '_proc').
Checks, using clients.csv and rounds.csv:
  1. net_download_s and net_upload_s match bytes*8/bandwidth + delay for every client and round
  2. slow_extra_s = (factor-1) x compute time for the slow clients and about 0 for the others
  3. the measured round time is not shorter than the slowest client's predicted time
     (the extra time above it is evaluation, aggregation and message passing: printed for information)
Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.network import NetworkEmulator, parse_condition, transfer_seconds


def condition_from_run(run_dir: Path) -> str:
    cond = run_dir.name.split("__")[-1]
    return cond[:-5] if cond.endswith("_proc") else cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", default=None, help="override the condition read from the folder name")
    args = ap.parse_args()

    run = Path(args.run)
    cond = args.condition or condition_from_run(run)
    cfg = parse_condition(cond)
    clients = pd.read_csv(run / "clients.csv")
    rounds = pd.read_csv(run / "rounds.csv")
    need = ["net_download_s", "slow_extra_s", "net_upload_s"]
    missing = [c for c in need if c not in clients.columns]
    if missing:
        print(f"FAIL clients.csv has no columns {missing}: the logging edit in run_process.py is not done yet")
        sys.exit(1)

    emu = NetworkEmulator(cfg, int(clients["client_id"].max()) + 1)
    print(f"run      : {run}\ncondition: {cfg.describe()}\nslow ids : {list(emu.slow_ids)}\n")

    def tol(x: float) -> float:
        return 0.05 * x + 0.05

    compute = clients["local_train_s"] + clients["snapshot_opt_s"].fillna(0.0)
    exp_down = clients["download_bytes"].apply(lambda b: transfer_seconds(int(b), cfg.bandwidth_mbps, cfg.delay_ms))
    exp_up = clients["upload_bytes"].apply(lambda b: transfer_seconds(int(b), cfg.bandwidth_mbps, cfg.delay_ms))
    exp_extra = [emu.extra_seconds(int(c), float(t)) for c, t in zip(clients["client_id"], compute)]
    exp_extra = pd.Series(exp_extra, index=clients.index)

    ok_down = bool(((clients["net_download_s"] - exp_down).abs() <= exp_down.apply(tol)).all())
    ok_up = bool(((clients["net_upload_s"] - exp_up).abs() <= exp_up.apply(tol)).all())
    ok_extra = bool(((clients["slow_extra_s"] - exp_extra).abs() <= exp_extra.apply(tol)).all())

    # predicted round time = slowest client (formula from the plan, Step 11), per round
    clients = clients.assign(pred=[
        emu.predicted_client_seconds(int(c), int(d), int(u), float(t))
        for c, d, u, t in zip(clients["client_id"], clients["download_bytes"], clients["upload_bytes"], compute)])
    pred_round = clients.groupby("round")["pred"].max()
    merged = rounds.set_index("round").join(pred_round)
    over = merged["round_wall_s"] - merged["pred"]
    ok_round = bool((over >= -0.1).all())

    show = merged[["round_wall_s", "pred"]].assign(overhead_s=over).round(3)
    print(show.rename(columns={"pred": "predicted_round_s"}).to_string(), "\n")
    results = [
        ("download pauses match bytes*8/bandwidth + delay", ok_down),
        ("upload pauses match bytes*8/bandwidth + delay", ok_up),
        ("slow clients pause (factor-1) x compute time, others about 0", ok_extra),
        ("measured round time >= predicted slowest-client time", ok_round),
    ]
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\ninfo: overhead per round (evaluation, aggregation, messages) = {over.mean():.2f} s on average")
    all_ok = all(ok for _, ok in results)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
