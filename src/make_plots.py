"""Step 13 (E1): build FedAvg vs OS-Fed comparison plots and tables from logs.

Reads the rounds.csv files that run_single.py wrote (never from memory) and
produces, under results/E1_baseline/:
  * E1_test_acc_vs_round_A.png        accuracy against communication round
  * E1_test_acc_vs_upload_A.png       accuracy against cumulative UPLOAD bytes
  * E1_summary_A.csv                  one row per method (acc + bytes + rounds-to-target)
  * E1_paper_reference.csv            the paper's Table 2 numbers, kept SEPARATE

Run from the project root:
    python -m src.make_plots
    python -m src.make_plots --target 0.40
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import project_root

# Which runs make up experiment E1. (label, log-folder glob, line style)
RUNS = [
    ("FedAvg (wavg)",        "*_A_fedavg_cnn_noniid_wavg_s1__net0", "-"),
    ("OS-Fed g4 (ec0)",      "*_A_osfed_cnn_noniid_g4ec0_s1__net0", "-"),
    ("OS-Fed g4 (ec1, diverged)", "*_A_osfed_cnn_noniid_g4ec1_s1__net0", "--"),
]

# Paper Table 2 (CIFAR-10, ResNet-18). Kept SEPARATE - NOT our results.
PAPER_TABLE2 = [
    ("FedAvg (paper, ResNet-18)",  0.476, 447.2),
    ("OS-Fed 2x2 (paper)",         0.422, 0.12),
    ("OS-Fed 4x4 (paper)",         0.449, 0.12),
]


def find_run(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.glob(f"logs/{pattern}"))
    return hits[-1] if hits else None


def rounds_to_target(df: pd.DataFrame, target: float) -> str:
    hit = df[df["test_acc"] >= target]
    return str(int(hit["round"].iloc[0])) if not hit.empty else f">{int(df['round'].max())}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.40,
                    help="target accuracy for the rounds-to-target column")
    args = ap.parse_args()

    root = project_root()
    out = root / "results" / "E1_baseline"
    out.mkdir(parents=True, exist_ok=True)

    loaded = []  # (label, df, style)
    for label, pattern, style in RUNS:
        run_dir = find_run(root, pattern)
        if run_dir is None:
            print(f"skip (no run found): {label}")
            continue
        df = pd.read_csv(run_dir / "rounds.csv")
        loaded.append((label, df, style))
        print(f"loaded {label}: {run_dir.name} ({len(df)} rounds)")

    if not loaded:
        raise SystemExit("No E1 runs found in logs/. Run FedAvg and OS-Fed first.")

    # --- Figure 1: accuracy vs round ---
    plt.figure(figsize=(7, 4.5))
    for label, df, style in loaded:
        plt.plot(df["round"], df["test_acc"], style, label=label)
    plt.xlabel("communication round")
    plt.ylabel("test accuracy")
    plt.title("E1: FedAvg vs OS-Fed (small CNN, CIFAR-10 non-IID)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "E1_test_acc_vs_round_A.png", dpi=150)
    plt.close()

    # --- Figure 2: accuracy vs cumulative UPLOAD (MB), log x ---
    plt.figure(figsize=(7, 4.5))
    for label, df, style in loaded:
        cum_up_mb = df["upload_bytes_total"].cumsum() / 1e6
        plt.plot(cum_up_mb, df["test_acc"], style, label=label)
    plt.xscale("log")
    plt.xlabel("cumulative upload (MB, all clients) - log scale")
    plt.ylabel("test accuracy")
    plt.title("E1: accuracy vs upload cost (lower-right is better)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "E1_test_acc_vs_upload_A.png", dpi=150)
    plt.close()

    # --- Summary table ---
    summary_path = out / "E1_summary_A.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "rounds", "final_test_acc", "best_test_acc",
            "upload_MB_per_round_total", "download_MB_per_round_total",
            "total_upload_MB", f"rounds_to_{args.target:.2f}",
        ])
        for label, df, _ in loaded:
            w.writerow([
                label,
                int(df["round"].max()),
                round(float(df["test_acc"].iloc[-1]), 4),
                round(float(df["test_acc"].max()), 4),
                round(float(df["upload_bytes_total"].mean()) / 1e6, 4),
                round(float(df["download_bytes_total"].mean()) / 1e6, 4),
                round(float(df["upload_bytes_total"].sum()) / 1e6, 2),
                rounds_to_target(df, args.target),
            ])

    # --- Paper reference (kept separate) ---
    ref_path = out / "E1_paper_reference.csv"
    with open(ref_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "paper_acc", "paper_MB_per_round"])
        for name, acc, mb in PAPER_TABLE2:
            w.writerow([name, acc, mb])

    print("\nsaved:")
    for p in ["E1_test_acc_vs_round_A.png", "E1_test_acc_vs_upload_A.png",
              "E1_summary_A.csv", "E1_paper_reference.csv"]:
        print("  ", out / p)

    print("\n=== E1 summary (ours) ===")
    print(pd.read_csv(summary_path).to_string(index=False))
    print("\n=== paper Table 2 (reference, NOT ours) ===")
    print(pd.read_csv(ref_path).to_string(index=False))


if __name__ == "__main__":
    main()
