"""
visualize.py — Generate plots from eval_results.csv.

Produces:
  results/robustness_curve.png  — upright fraction & success rate vs mismatch
  results/return_vs_mismatch.png — mean return with error bars
  results/heatmap.png            — upright fraction heatmap across all conditions

Usage:
  python visualize.py
  python visualize.py --csv results/eval_results.csv
"""
from __future__ import annotations
import argparse, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)

COLORS = {"nominal": "#2196F3", "positive": "#4CAF50",
          "negative": "#F44336", "random":   "#FF9800"}
LABELS = {"nominal": "Nominal (0%)", "positive": "All params +ε",
          "negative": "All params −ε", "random":  "Random ±ε"}


def plot_robustness_curve(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Policy Robustness vs Simulator-to-Real Mismatch",
                 fontsize=14, fontweight="bold")

    configs = [
        ("upright_fraction", "Fraction of Steps Upright"),
        ("success_rate",     "Success Rate  (≥50 % upright)"),
    ]
    for ax, (metric, ylabel) in zip(axes, configs):
        for direction, color in COLORS.items():
            sub = df[df["direction"] == direction].sort_values("mismatch_pct")
            if sub.empty:
                continue
            ax.plot(sub["mismatch_pct"], sub[metric], "o-",
                    color=color, label=LABELS[direction], linewidth=2, markersize=7)

        ax.axhline(0.50, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                   label="50 % baseline")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_xlabel("Mismatch Level ε (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(sorted(df["mismatch_pct"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = "results/robustness_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_return(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for direction, color in COLORS.items():
        sub = df[df["direction"] == direction].sort_values("mismatch_pct")
        if sub.empty:
            continue
        yerr = sub["std_return"].values / np.sqrt(100)
        ax.errorbar(sub["mismatch_pct"], sub["mean_return"], yerr=yerr,
                    fmt="o-", color=color, label=LABELS[direction],
                    linewidth=2, markersize=7, capsize=4)

    ax.set_title("Mean Episodic Return vs Mismatch Level",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mismatch Level ε (%)", fontsize=11)
    ax.set_ylabel("Mean Episodic Return", fontsize=11)
    ax.set_xticks(sorted(df["mismatch_pct"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = "results/return_vs_mismatch.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="direction", columns="mismatch_pct",
                           values="upright_fraction")
    order = [d for d in ["nominal", "positive", "negative", "random"]
             if d in pivot.index]
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Upright Fraction")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"±{c}%" if c > 0 else "0%" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Mismatch Level ε")
    ax.set_title("Upright Fraction — Heatmap by Condition",
                 fontsize=12, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt_color = "black" if 0.25 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=txt_color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = "results/heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def main(csv_path: str = "results/eval_results.csv"):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    plot_robustness_curve(df)
    plot_return(df)
    plot_heatmap(df)
    print("All plots saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/eval_results.csv")
    args = parser.parse_args()
    main(args.csv)
