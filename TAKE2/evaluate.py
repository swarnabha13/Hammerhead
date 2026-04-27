"""
evaluate.py  –  Evaluate Trained Policy Across Mismatch Levels
===============================================================
Loads a frozen, simulation-trained PPO policy and evaluates it across
a grid of simulator-to-real mismatch levels.

Mismatch is defined as a scalar multiplier applied to ALL varied physical
parameters simultaneously:
    p_eval = p_nominal * (1 + mismatch_level)

Evaluation levels: -0.20, -0.15, -0.10, -0.05, -0.02, 0.00,
                   +0.02, +0.05, +0.10, +0.15, +0.20

Success Criterion (documented in README.md)
-------------------------------------------
  An episode is "successful" if the agent holds the Acrobot near upright
  for the required consecutive balance window before the 500-step time limit.
  In Gymnasium terms: terminated=True (not merely truncated=True).

  Primary metric : Success Rate (%) — fraction of 100 episodes that succeed.
  Secondary metric: Mean Episode Return — average cumulative reward per episode.
  Tertiary metric : Mean Steps to Balance — mean steps in successful episodes.

Usage
-----
# Evaluate latest checkpoint with default settings
python evaluate.py

# Evaluate specific checkpoint
python evaluate.py --checkpoint checkpoints/my_run.pt

# Evaluate nominal-only baseline (no DR) for comparison
python evaluate.py --checkpoint checkpoints/baseline.pt --label "No DR"

# Change number of evaluation episodes
python evaluate.py --num-episodes 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

sys.path.insert(0, os.path.dirname(__file__))
from envs.randomized_acrobot import (
    MAX_EPISODE_STEPS,
    NOMINAL_PARAMS,
    OBSERVATION_DIM,
    RandomizedAcrobotEnv,
    TORQUE_VALUES,
    VARIED_PARAMS,
)
from train import ActorCritic  # reuse network definition


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy across mismatch levels"
    )
    parser.add_argument("--checkpoint",   type=str,   default="checkpoints/latest.pt",
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--num-episodes", type=int,   default=100,
                        help="Number of evaluation episodes per mismatch level")
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--cuda",         action="store_true", default=False)
    parser.add_argument("--label",        type=str,   default=None,
                        help="Label for plots/CSV (defaults to checkpoint filename)")
    parser.add_argument("--out-dir",      type=str,   default="results",
                        help="Directory to save CSV and PNG results")
    parser.add_argument("--no-plots",     action="store_true", default=False,
                        help="Skip matplotlib plot generation")
    return parser.parse_args()


# ============================================================
# Mismatch levels to evaluate
# ============================================================

MISMATCH_LEVELS = [-0.20, -0.15, -0.10, -0.05, -0.02, 0.00,
                    0.02,  0.05,  0.10,  0.15,  0.20]


# ============================================================
# Load model
# ============================================================

def load_agent(checkpoint_path: str, device: torch.device) -> tuple[ActorCritic, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate a checkpoint."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_obs_dim = ckpt.get("obs_dim", OBSERVATION_DIM)
    if ckpt_obs_dim != OBSERVATION_DIM:
        raise ValueError(
            f"Checkpoint obs_dim={ckpt_obs_dim}, but the current two-link "
            f"balance environment expects obs_dim={OBSERVATION_DIM}. "
            "Retrain the policy with the current code before evaluating."
        )
    expected_action_dim = len(TORQUE_VALUES)
    ckpt_action_dim = ckpt.get("action_dim", expected_action_dim)
    if ckpt_action_dim != expected_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={ckpt_action_dim}, but the current two-link "
            f"balance environment expects action_dim={expected_action_dim}. "
            "Retrain the policy with the current code before evaluating."
        )
    agent = ActorCritic(
        obs_dim    = OBSERVATION_DIM,
        action_dim = expected_action_dim,
    ).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent, ckpt


# ============================================================
# Evaluate one mismatch level
# ============================================================

def evaluate_mismatch(
    agent:           ActorCritic,
    mismatch:        float,
    num_episodes:    int,
    seed:            int,
    device:          torch.device,
) -> dict:
    """
    Run `num_episodes` episodes at a fixed mismatch level and return metrics.

    Returns
    -------
    dict with keys:
        mismatch_pct   : mismatch level as percentage (e.g. 10.0 for +10%)
        success_rate   : fraction of episodes that balanced before timeout
        mean_return    : mean episodic return
        std_return     : std of episodic return
        mean_steps     : mean episode length
        mean_steps_success : mean steps only for balanced episodes (NaN if none)
        n_episodes     : number of episodes run
        active_params  : dict of effective parameter values (at this mismatch)
    """
    env = RandomizedAcrobotEnv(
        dr_range       = 0.0,           # No randomization – fixed mismatch only
        fixed_mismatch = mismatch,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    np.random.seed(seed)
    torch.manual_seed(seed)

    returns          : list[float] = []
    lengths          : list[int]   = []
    successes        : list[bool]  = []
    success_lengths  : list[int]   = []
    target_reaches   : list[bool]  = []
    max_hold_steps   : list[int]   = []

    obs, _ = env.reset(seed=seed)

    ep_count = 0
    ep_target_reached = False
    ep_max_hold_steps = 0
    while ep_count < num_episodes:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.get_deterministic_action(obs_tensor)
        obs, _, terminated, truncated, info = env.step(action.item())
        ep_target_reached = ep_target_reached or bool(info.get("target_reached", False))
        ep_max_hold_steps = max(ep_max_hold_steps, int(info.get("balance_steps", 0)))

        if terminated or truncated:
            ep_count += 1
            ep_info = info.get("episode", {})
            ep_ret  = float(ep_info.get("r", 0.0))
            ep_len  = int(ep_info.get("l", 0))
            returns.append(ep_ret)
            lengths.append(ep_len)
            success = bool(terminated)
            successes.append(success)
            target_reaches.append(ep_target_reached)
            max_hold_steps.append(ep_max_hold_steps)
            if success:
                success_lengths.append(ep_len)
            obs, _ = env.reset()
            ep_target_reached = False
            ep_max_hold_steps = 0

    env.close()

    success_rate = np.mean(successes)
    mean_return  = np.mean(returns)
    std_return   = np.std(returns)
    mean_steps   = np.mean(lengths)
    mean_steps_success = np.mean(success_lengths) if success_lengths else float("nan")
    target_reach_rate = np.mean(target_reaches)
    mean_max_hold_steps = np.mean(max_hold_steps)

    # Record effective parameter values at this mismatch level
    active_params = {k: NOMINAL_PARAMS[k] * (1 + mismatch) if k in VARIED_PARAMS
                     else NOMINAL_PARAMS[k]
                     for k in NOMINAL_PARAMS}

    return {
        "mismatch_pct"       : mismatch * 100,
        "success_rate"       : success_rate * 100,
        "mean_return"        : mean_return,
        "std_return"         : std_return,
        "mean_steps"         : mean_steps,
        "mean_steps_success" : mean_steps_success,
        "target_reach_rate"  : target_reach_rate * 100,
        "mean_max_hold_steps": mean_max_hold_steps,
        "n_episodes"         : num_episodes,
        "active_params"      : active_params,
    }


# ============================================================
# Plotting
# ============================================================

def plot_results(df: pd.DataFrame, out_dir: str, label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("  matplotlib not available – skipping plots")
        return

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Robust Acrobot Evaluation – {label}", fontweight="bold", fontsize=13)

    x       = df["mismatch_pct"].values
    colors  = ["#d62728" if v < 0 else "#1f77b4" if v > 0 else "#2ca02c" for v in x]

    # ---- Plot 1: Success Rate ----
    ax = axes[0]
    bars = ax.bar(x, df["success_rate"], width=1.8, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs. Mismatch")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, val in zip(bars, df["success_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    # ---- Plot 2: Mean Return ----
    ax = axes[1]
    ax.plot(x, df["mean_return"], "o-", color="#1f77b4", linewidth=2, markersize=6, label="Mean")
    ax.fill_between(x,
                    df["mean_return"] - df["std_return"],
                    df["mean_return"] + df["std_return"],
                    alpha=0.2, color="#1f77b4", label="±1 SD")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="No mismatch")
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Episodic Return")
    ax.set_title("Mean Return vs. Mismatch")
    ax.legend(fontsize=9)

    # ---- Plot 3: Mean Steps to Solve ----
    ax = axes[2]
    ax.plot(x, df["mean_steps_success"], "s-", color="#2ca02c", linewidth=2,
            markersize=6, label="Steps (success only)")
    ax.plot(x, df["mean_steps"], "^--", color="#ff7f0e", linewidth=1.5,
            markersize=5, alpha=0.7, label="Steps (all episodes)")
    ax.axhline(y=500, color="red", linestyle=":", alpha=0.5, label="Max steps")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Steps to Balance vs. Mismatch")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"eval_{label.replace(' ', '_')}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


# ============================================================
# Main
# ============================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent, ckpt = load_agent(args.checkpoint, device)
    train_args  = ckpt.get("args", {})
    dr_range    = train_args.get("dr_range", "unknown")
    label       = args.label or os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"  Model       : {ckpt.get('run_name', 'unknown')}")
    print(f"  Trained DR  : ±{float(dr_range)*100:.0f}%" if dr_range != "unknown" else "  Trained DR  : unknown")
    print(f"  Global step : {ckpt.get('global_step', 'unknown'):,}" if isinstance(ckpt.get('global_step'), int) else "")
    print(f"  Eval label  : {label}")
    print(f"  Episodes    : {args.num_episodes} per level")
    print(f"  Mismatch levels: {[f'{m*100:+.0f}%' for m in MISMATCH_LEVELS]}")
    print()

    # ---- Run evaluations ----
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []

    for mismatch in MISMATCH_LEVELS:
        t0 = time.time()
        res = evaluate_mismatch(
            agent        = agent,
            mismatch     = mismatch,
            num_episodes = args.num_episodes,
            seed         = args.seed,
            device       = device,
        )
        elapsed = time.time() - t0
        all_results.append(res)
        print(
            f"  Mismatch {res['mismatch_pct']:+6.1f}% | "
            f"Success {res['success_rate']:6.1f}% | "
            f"Target {res['target_reach_rate']:6.1f}% | "
            f"Hold {res['mean_max_hold_steps']:5.1f} | "
            f"Return {res['mean_return']:7.1f} ± {res['std_return']:.1f} | "
            f"Steps {res['mean_steps']:5.1f} | "
            f"({elapsed:.1f}s)"
        )

    # ---- Build DataFrame ----
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "active_params"}
        for r in all_results
    ])

    # ---- Print summary table ----
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY  –  {label}")
    print(f"{'='*70}")
    print(df[["mismatch_pct", "success_rate", "mean_return", "std_return",
              "mean_steps", "mean_steps_success", "target_reach_rate",
              "mean_max_hold_steps"]].to_string(index=False,
              float_format=lambda x: f"{x:.2f}"))

    # ---- Robustness score ----
    nom_success = df.loc[df["mismatch_pct"] == 0.0, "success_rate"].values
    nom_success = nom_success[0] if len(nom_success) > 0 else 100.0
    worst_success = df["success_rate"].min()
    robustness_drop = nom_success - worst_success
    print(f"\n  Nominal success rate : {nom_success:.1f}%")
    print(f"  Worst-case success   : {worst_success:.1f}%  "
          f"(at mismatch = {df.loc[df['success_rate'].idxmin(), 'mismatch_pct']:+.1f}%)")
    print(f"  Max robustness drop  : {robustness_drop:.1f} percentage points")
    print(f"{'='*70}\n")

    # ---- Save CSV ----
    csv_path = os.path.join(args.out_dir, f"eval_{label.replace(' ', '_')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")

    # ---- Plots ----
    if not args.no_plots:
        plot_results(df, args.out_dir, label)

    return df


if __name__ == "__main__":
    main()
