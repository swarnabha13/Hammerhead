"""
compare_runs.py  –  Compare Multiple Policies Side-by-Side
===========================================================
Evaluates two policies (e.g. DR-trained vs. no-DR baseline) and produces
a comparison plot showing how robustness differs.

Usage
-----
# Compare DR policy vs. baseline
python compare_runs.py \
    --checkpoints checkpoints/ppo_robust_acrobot*.pt checkpoints/ppo_no_dr*.pt \
    --labels "PPO+DR (±10%)" "PPO (no DR)"

# Use the two most recent checkpoints
python compare_runs.py --auto
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import torch
from evaluate import load_agent, evaluate_mismatch, MISMATCH_LEVELS
from train import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=str,
                        help="Paths to checkpoint .pt files")
    parser.add_argument("--labels",      nargs="+", type=str,
                        help="Labels for each checkpoint (same order)")
    parser.add_argument("--auto",        action="store_true",
                        help="Auto-detect all .pt files in checkpoints/")
    parser.add_argument("--num-episodes",type=int,  default=100)
    parser.add_argument("--seed",        type=int,  default=0)
    parser.add_argument("--out-dir",     type=str,  default="results")
    return parser.parse_args()


def compare(args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not available – skipping plots")
        return

    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Auto mode is handy when comparing several local training attempts.
    if args.auto:
        ckpt_paths = sorted(glob.glob("checkpoints/*.pt"))
        ckpt_paths = [p for p in ckpt_paths if "latest" not in p]
    else:
        ckpt_paths = args.checkpoints or []

    if not ckpt_paths:
        print("No checkpoints found. Train first with train.py")
        return

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in ckpt_paths]
    print(f"Comparing {len(ckpt_paths)} policies:\n")
    for p, l in zip(ckpt_paths, labels):
        print(f"  [{l}] → {p}")
    print()

    all_dfs = {}
    for ckpt_path, label in zip(ckpt_paths, labels):
        agent, _ = load_agent(ckpt_path, device)
        results  = []
        for mismatch in MISMATCH_LEVELS:
            res = evaluate_mismatch(agent, mismatch, args.num_episodes, args.seed, device)
            results.append(res)
            print(f"  [{label}] Mismatch {res['mismatch_pct']:+6.1f}% | "
                  f"Hold {res['mean_max_hold_steps']:.1f} steps")
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "active_params"}
                           for r in results])
        all_dfs[label] = df
        csv_path = os.path.join(args.out_dir, f"eval_{label.replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)
    print()

    # Comparison plot
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Policy Robustness Comparison", fontweight="bold", fontsize=13)

    x = MISMATCH_LEVELS

    # Primary metric: longest continuous hold.
    ax = axes[0]
    for (label, df), color in zip(all_dfs.items(), colors):
        ax.plot([m*100 for m in x], df["mean_max_hold_steps"], "o-",
                color=color, linewidth=2, markersize=6, label=label)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Max Consecutive Hold Steps")
    ax.set_title("Longest Hold vs. Mismatch")
    ax.legend(fontsize=9)

    # Return is shown as a diagnostic, not as the main success metric.
    ax = axes[1]
    for (label, df), color in zip(all_dfs.items(), colors):
        ax.plot([m*100 for m in x], df["mean_return"], "s--",
                color=color, linewidth=2, markersize=5, label=label)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Mean Episodic Return")
    ax.set_title("Return vs. Mismatch")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Comparison plot saved: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    compare(args)
