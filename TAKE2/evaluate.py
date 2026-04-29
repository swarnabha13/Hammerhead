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

Balance Objective (documented in README.md)
-------------------------------------------
  The policy is evaluated by the longest consecutive streak spent near upright.
  The near-upright region keeps both links within 10 degrees of vertical and
  applies velocity limits so high-speed swing-throughs are not counted.

  Primary metric : Mean Max Hold Steps — average longest hold streak.
  Secondary metric: Hold Score (%) — mean max hold as a fraction of episode length.
  Tertiary metrics: Upright Time (%) and Balanced Time (%) across each episode.

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
from train import ActorCritic  # keep evaluation on the same network class


# CLI setup

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
    parser.add_argument("--controller",   choices=["policy", "hybrid", "mpc"],
                        default="policy",
                        help="Action source: learned policy, MPC expert, or hybrid policy+MPC stabilizer")
    parser.add_argument("--balance-reset-prob", type=float, default=0.0,
                        help="Probability of starting each eval episode near upright")
    return parser.parse_args()


# Fixed mismatch levels used in the report.

MISMATCH_LEVELS = [-0.20, -0.15, -0.10, -0.05, -0.02, 0.00,
                    0.02,  0.05,  0.10,  0.15,  0.20]


# Checkpoint loading

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


# Single mismatch evaluation

def evaluate_mismatch(
    agent:           ActorCritic,
    mismatch:        float,
    num_episodes:    int,
    seed:            int,
    device:          torch.device,
    controller:      str = "policy",
    balance_reset_prob: float = 0.0,
) -> dict:
    """
    Run `num_episodes` episodes at a fixed mismatch level and return metrics.

    Returns
    -------
    dict with keys:
        mismatch_pct   : mismatch level as percentage (e.g. 10.0 for +10%)
        hold_score_pct : mean longest hold streak as a percent of episode length
        mean_return    : mean episodic return
        std_return     : std of episodic return
        mean_steps     : mean episode length
        mean_max_hold_steps : mean longest consecutive near-upright streak
        n_episodes     : number of episodes run
        active_params  : dict of effective parameter values (at this mismatch)
    """
    env = RandomizedAcrobotEnv(
        dr_range       = 0.0,           # fixed mismatch only; no randomization
        fixed_mismatch = mismatch,
        balance_reset_prob = balance_reset_prob,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    np.random.seed(seed)
    torch.manual_seed(seed)

    returns          : list[float] = []
    lengths          : list[int]   = []
    target_reaches   : list[bool]  = []
    max_hold_steps   : list[int]   = []
    upright_time_pct  : list[float] = []
    balance_time_pct  : list[float] = []

    obs, _ = env.reset(seed=seed)

    ep_count = 0
    ep_target_reached = False
    ep_max_hold_steps = 0
    ep_upright_steps = 0
    ep_balanced_steps = 0
    while ep_count < num_episodes:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if controller == "mpc":
                action_item = env.unwrapped.expert_action()
            elif controller == "hybrid" and env.unwrapped.should_use_balance_controller():
                action_item = env.unwrapped.expert_action()
            else:
                action_item = int(agent.get_deterministic_action(obs_tensor).item())
        obs, _, terminated, truncated, info = env.step(action_item)
        ep_target_reached = ep_target_reached or bool(info.get("target_reached", False))
        ep_max_hold_steps = max(ep_max_hold_steps, int(info.get("balance_steps", 0)))
        ep_upright_steps += int(info.get("upright", False))
        ep_balanced_steps += int(info.get("balanced", False))

        if terminated or truncated:
            ep_count += 1
            ep_info = info.get("episode", {})
            ep_ret  = float(ep_info.get("r", 0.0))
            ep_len  = int(ep_info.get("l", 0))
            returns.append(ep_ret)
            lengths.append(ep_len)
            target_reaches.append(ep_target_reached)
            max_hold_steps.append(ep_max_hold_steps)
            upright_time_pct.append(100.0 * ep_upright_steps / max(1, ep_len))
            balance_time_pct.append(100.0 * ep_balanced_steps / max(1, ep_len))
            obs, _ = env.reset()
            ep_target_reached = False
            ep_max_hold_steps = 0
            ep_upright_steps = 0
            ep_balanced_steps = 0

    env.close()

    mean_return  = np.mean(returns)
    std_return   = np.std(returns)
    mean_steps   = np.mean(lengths)
    target_reach_rate = np.mean(target_reaches)
    mean_max_hold_steps = np.mean(max_hold_steps)
    hold_score_pct = 100.0 * mean_max_hold_steps / MAX_EPISODE_STEPS
    mean_upright_time_pct = np.mean(upright_time_pct)
    mean_balance_time_pct = np.mean(balance_time_pct)

    # Store the effective parameters with the row for easier debugging later.
    active_params = {k: NOMINAL_PARAMS[k] * (1 + mismatch) if k in VARIED_PARAMS
                     else NOMINAL_PARAMS[k]
                     for k in NOMINAL_PARAMS}

    return {
        "mismatch_pct"       : mismatch * 100,
        "hold_score_pct"     : hold_score_pct,
        "mean_return"        : mean_return,
        "std_return"         : std_return,
        "mean_steps"         : mean_steps,
        "target_reach_rate"  : target_reach_rate * 100,
        "mean_max_hold_steps": mean_max_hold_steps,
        "mean_upright_time_pct": mean_upright_time_pct,
        "mean_balance_time_pct": mean_balance_time_pct,
        "n_episodes"         : num_episodes,
        "active_params"      : active_params,
    }


# Plotting

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

    # Plot 1: longest continuous hold.
    ax = axes[0]
    bars = ax.bar(x, df["mean_max_hold_steps"], width=1.8, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=MAX_EPISODE_STEPS, color="gray", linestyle="--", alpha=0.4, label="Full episode")
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Max Consecutive Hold Steps")
    ax.set_title("Longest Hold vs. Mismatch")
    ax.set_ylim(0, MAX_EPISODE_STEPS * 1.1)
    for bar, val in zip(bars, df["mean_max_hold_steps"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    # Plot 2: shaped return, mostly useful as a diagnostic.
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

    # Plot 3: time spent in the relaxed and strict balance regions.
    ax = axes[2]
    ax.plot(x, df["mean_upright_time_pct"], "s-", color="#2ca02c", linewidth=2,
            markersize=6, label="Link 2 within 10 deg")
    ax.plot(x, df["mean_balance_time_pct"], "^--", color="#ff7f0e", linewidth=1.5,
            markersize=5, alpha=0.8, label="Full balance")
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, label="Perfect")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mismatch Level (%)")
    ax.set_ylabel("Episode Time (%)")
    ax.set_title("Upright Time vs. Mismatch")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"eval_{label.replace(' ', '_')}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


# Main

def main():
    args   = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent, ckpt = load_agent(args.checkpoint, device)
    train_args  = ckpt.get("args", {})
    dr_range    = train_args.get("dr_range", "unknown")
    label       = args.label or os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"  Model       : {ckpt.get('run_name', 'unknown')}")
    print(f"  Trained DR  : ±{float(dr_range)*100:.0f}%" if dr_range != "unknown" else "  Trained DR  : unknown")
    print(f"  Global step : {ckpt.get('global_step', 'unknown'):,}" if isinstance(ckpt.get('global_step'), int) else "")
    print(f"  Eval label  : {label}")
    print(f"  Controller  : {args.controller}")
    print(f"  Balance reset: {args.balance_reset_prob*100:.0f}%")
    print(f"  Episodes    : {args.num_episodes} per level")
    print(f"  Mismatch levels: {[f'{m*100:+.0f}%' for m in MISMATCH_LEVELS]}")
    print()

    # Run evaluations
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
            controller   = args.controller,
            balance_reset_prob = args.balance_reset_prob,
        )
        elapsed = time.time() - t0
        all_results.append(res)
        print(
            f"  Mismatch {res['mismatch_pct']:+6.1f}% | "
            f"Hold {res['mean_max_hold_steps']:6.1f} | "
            f"HoldScore {res['hold_score_pct']:5.1f}% | "
            f"Target {res['target_reach_rate']:6.1f}% | "
            f"Upright {res['mean_upright_time_pct']:5.1f}% | "
            f"BalTime {res['mean_balance_time_pct']:5.1f}% | "
            f"Return {res['mean_return']:7.1f} ± {res['std_return']:.1f} | "
            f"Steps {res['mean_steps']:5.1f} | "
            f"({elapsed:.1f}s)"
        )

    # Build the flat table saved to CSV.
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "active_params"}
        for r in all_results
    ])

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY  –  {label}")
    print(f"{'='*70}")
    print(df[["mismatch_pct", "hold_score_pct", "mean_return", "std_return",
              "mean_steps", "target_reach_rate",
              "mean_max_hold_steps", "mean_upright_time_pct",
              "mean_balance_time_pct"]].to_string(index=False,
              float_format=lambda x: f"{x:.2f}"))

    # Simple robustness summary: nominal hold minus worst-case hold.
    nom_hold = df.loc[df["mismatch_pct"] == 0.0, "mean_max_hold_steps"].values
    nom_hold = nom_hold[0] if len(nom_hold) > 0 else 0.0
    worst_hold = df["mean_max_hold_steps"].min()
    robustness_drop = nom_hold - worst_hold
    print(f"\n  Nominal mean max hold : {nom_hold:.1f} steps")
    print(f"  Worst-case max hold   : {worst_hold:.1f} steps  "
          f"(at mismatch = {df.loc[df['mean_max_hold_steps'].idxmin(), 'mismatch_pct']:+.1f}%)")
    print(f"  Max robustness drop   : {robustness_drop:.1f} hold steps")
    print(f"{'='*70}\n")

    # Save CSV
    csv_path = os.path.join(args.out_dir, f"eval_{label.replace(' ', '_')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")

    # Plots
    if not args.no_plots:
        plot_results(df, args.out_dir, label)

    return df


if __name__ == "__main__":
    main()
