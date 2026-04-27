"""
evaluate.py — Evaluate swing-up + balance policy across mismatch levels
=======================================================================
Success criterion (updated for balance task):
  An episode is "successful" if the policy holds BOTH links upright
  (tip_height ≥ 1.8) for ≥ 100 consecutive steps before the 1000-step limit.

Additional metrics vs. swing-up-only:
  - mean_balance_steps : avg consecutive upright steps across all episodes
  - max_balance_steps  : best sustained balance in any episode
  - upright_fraction   : fraction of all steps spent in the upright region

Mismatch levels: -20%, -10%, -5%, -2%, 0%, +2%, +5%, +10%, +20%

Usage
-----
    python evaluate.py
    python evaluate.py --model-path results/ppo_acrobot.pth --n-episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from common import Agent, AcrobotBalanceEnv, DomainRandomizedAcrobot
from common import UPRIGHT_HEIGHT, BALANCE_STEPS_WIN

MISMATCH_LEVELS = [-0.20, -0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10, 0.20]


# ── Single-level evaluation ────────────────────────────────────────────────────
def evaluate_policy(
    agent: Agent,
    mismatch_level: float,
    n_episodes: int = 50,
    seed: int = 0,
    device: str = "cpu",
    max_episode_steps: int = 1000,
) -> Dict:
    env = gym.make("Acrobot-v1", max_episode_steps=max_episode_steps)
    env = AcrobotBalanceEnv(env)
    env = DomainRandomizedAcrobot(
        env, randomize=False, mismatch_level=mismatch_level
    )

    returns:         List[float] = []
    successes:       List[int]   = []
    balance_steps:   List[int]   = []
    upright_fracs:   List[float] = []

    agent.eval()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done          = False
        ep_return     = 0.0
        ep_steps      = 0
        max_consec    = 0
        cur_consec    = 0
        total_upright = 0
        terminated_   = False

        while not done:
            action = agent.act_single(obs, device=device)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_steps  += 1
            done        = terminated or truncated
            terminated_ = bool(terminated)

            th = AcrobotBalanceEnv.tip_height(obs)
            if th >= UPRIGHT_HEIGHT:
                cur_consec    += 1
                total_upright += 1
                max_consec     = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        returns.append(ep_return)
        successes.append(int(max_consec >= BALANCE_STEPS_WIN))
        balance_steps.append(max_consec)
        upright_fracs.append(total_upright / ep_steps if ep_steps > 0 else 0.0)

    env.close()
    return {
        "mismatch_level":    mismatch_level,
        "success_rate":      float(np.mean(successes)),
        "mean_return":       float(np.mean(returns)),
        "std_return":        float(np.std(returns)),
        "mean_balance_steps": float(np.mean(balance_steps)),
        "max_balance_steps": int(np.max(balance_steps)),
        "mean_upright_frac": float(np.mean(upright_fracs)),
        "n_episodes":        n_episodes,
        "all_returns":       returns,
        "all_successes":     successes,
        "all_balance_steps": balance_steps,
    }


# ── Full sweep ─────────────────────────────────────────────────────────────────
def evaluate_all(
    model_path: str,
    n_episodes: int = 50,
    seed: int = 0,
    save_dir: str = "./results",
    mismatch_levels: Optional[List[float]] = None,
) -> List[Dict]:
    if mismatch_levels is None:
        mismatch_levels = MISMATCH_LEVELS

    device    = "cpu"
    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim   = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]

    agent = Agent(obs_dim, n_actions).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])

    print(f"\n{'='*65}")
    print(f"  Swing-Up + Balance Evaluation: {model_path}")
    print(f"  Success = hold upright ≥ {BALANCE_STEPS_WIN} consecutive steps")
    print(f"  Episodes per level: {n_episodes}")
    print(f"{'='*65}")
    print(f"  {'Level':>8} | {'Success':>8} | {'Avg Bal':>8} | "
          f"{'MaxBal':>7} | {'Uprt%':>6} | {'Mean Ret':>9}")
    print(f"  {'-'*60}")

    results = []
    for level in mismatch_levels:
        r = evaluate_policy(
            agent, level, n_episodes=n_episodes, seed=seed, device=device)
        results.append(r)
        print(
            f"  {level:>+8.0%} | "
            f"{r['success_rate']:>7.1%} | "
            f"{r['mean_balance_steps']:>8.1f} | "
            f"{r['max_balance_steps']:>7d} | "
            f"{r['mean_upright_frac']:>5.1%} | "
            f"{r['mean_return']:>9.2f}"
        )

    os.makedirs(save_dir, exist_ok=True)
    slim = [{k: v for k, v in r.items()
             if k not in ("all_returns","all_successes","all_balance_steps")}
            for r in results]
    json_path = os.path.join(save_dir, "eval_results.json")
    with open(json_path, "w") as fh:
        json.dump(slim, fh, indent=2)
    print(f"\n  Results → {json_path}")

    plot_path = _plot_results(results, save_dir)
    print(f"  Plot    → {plot_path}\n")
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────
def _plot_results(results: List[Dict], save_dir: str) -> str:
    levels    = [r["mismatch_level"] * 100 for r in results]
    succ      = [r["success_rate"]   * 100 for r in results]
    bal_steps = [r["mean_balance_steps"]   for r in results]
    upr_frac  = [r["mean_upright_frac"] * 100 for r in results]
    mean_rets = [r["mean_return"]          for r in results]
    std_rets  = [r["std_return"]           for r in results]

    label_str = [f"{l:+.0f}%" for l in levels]
    x         = np.arange(len(levels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Acrobot Swing-Up + Balance — Sim-to-Real Robustness Evaluation",
        fontsize=14, fontweight="bold",
    )

    bar_colors = [
        "#2ecc71" if s >= 80 else "#f39c12" if s >= 40 else "#e74c3c"
        for s in succ
    ]

    # 1. Success rate
    ax = axes[0, 0]
    bars = ax.bar(x, succ, color=bar_colors, edgecolor="white", width=0.6, zorder=3)
    nom  = [r["mismatch_level"] for r in results]
    if 0.0 in nom:
        bars[nom.index(0.0)].set_edgecolor("black"); bars[nom.index(0.0)].set_linewidth(2)
    ax.axhline(80, color="green", ls="--", lw=1.2, label="80% target", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(label_str, rotation=45, ha="right")
    ax.set_ylabel("Success Rate (%)"); ax.set_title("Balance Success Rate")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, succ):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    # 2. Mean consecutive balance steps
    ax2 = axes[0, 1]
    ax2.bar(x, bal_steps, color="#9b59b6", edgecolor="white", width=0.6, zorder=3)
    ax2.axhline(BALANCE_STEPS_WIN, color="green", ls="--", lw=1.2,
                label=f"Success threshold ({BALANCE_STEPS_WIN} steps)", zorder=2)
    ax2.set_xticks(x); ax2.set_xticklabels(label_str, rotation=45, ha="right")
    ax2.set_ylabel("Avg. Consecutive Upright Steps")
    ax2.set_title("Mean Balance Duration")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    # 3. Upright fraction
    ax3 = axes[1, 0]
    ax3.bar(x, upr_frac, color="#e67e22", edgecolor="white", width=0.6, zorder=3)
    ax3.set_xticks(x); ax3.set_xticklabels(label_str, rotation=45, ha="right")
    ax3.set_ylabel("% Steps in Upright Region")
    ax3.set_title("Time Spent Upright (tip_height ≥ 1.8)")
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax3.grid(axis="y", alpha=0.3)

    # 4. Mean return
    ax4 = axes[1, 1]
    ax4.bar(x, mean_rets, yerr=std_rets, color="#3498db", edgecolor="white",
            width=0.6, capsize=4, error_kw={"elinewidth": 1.2}, zorder=3)
    ax4.set_xticks(x); ax4.set_xticklabels(label_str, rotation=45, ha="right")
    ax4.set_ylabel("Mean Episode Return")
    ax4.set_title("Mean Return ± 1 Std")
    ax4.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "eval_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_training_curve(history_path: str, save_dir: str) -> str:
    with open(history_path) as fh:
        hist = json.load(fh)
    ep_returns   = hist.get("episode_returns",   [])
    ep_successes = hist.get("episode_successes", [])
    if not ep_returns:
        return ""

    window = 100
    sm_ret  = np.convolve(ep_returns,   np.ones(window)/window, mode="valid")
    sm_succ = (np.convolve(ep_successes, np.ones(window)/window, mode="valid")
               * 100 if ep_successes else None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(ep_returns, color="#aab7c4", alpha=0.3, lw=0.6, label="Raw")
    ax1.plot(range(window-1, len(ep_returns)), sm_ret,
             color="#2980b9", lw=2, label=f"Smoothed (w={window})")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Training Curve — Acrobot Swing-Up + Balance (PPO + DR ±20%)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    if sm_succ is not None:
        ax2.fill_between(range(window-1, len(ep_successes)),
                         sm_succ, alpha=0.3, color="#2ecc71")
        ax2.plot(range(window-1, len(ep_successes)), sm_succ,
                 color="#27ae60", lw=2)
        ax2.axhline(80, color="green", ls="--", lw=1, label="80% target")
        ax2.set_ylabel("Balance Success Rate (%)")
        ax2.set_xlabel("Episode")
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="./results/ppo_acrobot.pth")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--save-dir",   type=str, default="./results")
    a = p.parse_args()

    results = evaluate_all(
        model_path=a.model_path,
        n_episodes=a.n_episodes,
        seed=a.seed,
        save_dir=a.save_dir,
    )
    history_path = os.path.join(a.save_dir, "training_history.json")
    if os.path.exists(history_path):
        c = _plot_training_curve(history_path, a.save_dir)
        if c:
            print(f"  Training curve → {c}\n")
