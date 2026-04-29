"""
evaluate.py — Evaluate a frozen PPO policy across mismatch levels.

For each mismatch level ε ∈ {0, 2, 5, 10, 20, 30}%:
  • Positive  (+ε): all randomised params × (1+ε)
  • Negative  (-ε): all randomised params × (1-ε)
  • Random   (±ε): each param sampled uniformly in [1-ε, 1+ε]

Metrics per condition:
  • mean_return        — average episodic return
  • upright_fraction   — fraction of timesteps with tip height > 1.0
  • success_rate       — fraction of episodes where upright_fraction ≥ 0.5
  • steps_upright_mean — average upright timesteps per episode

Usage:
  python evaluate.py --checkpoint checkpoints/NAME_best.pt
  python evaluate.py --checkpoint checkpoints/NAME_best.pt --n-episodes 200
"""
from __future__ import annotations

import argparse, os, time
import numpy as np
import pandas as pd
import torch

from envs.acrobot_custom import AcrobotBalanceEnv, build_eval_params, NOMINAL_PARAMS
from train import Agent

MISMATCH_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
DIRECTIONS      = ["positive", "negative", "random"]
SUCCESS_THRESH  = 0.50   # success here means at least half the episode was upright


# Load policy
def load_agent(checkpoint_path: str, device: torch.device) -> Agent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    env  = AcrobotBalanceEnv()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    env.close()
    agent = Agent(obs_dim, act_dim)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    step = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint: {checkpoint_path}  (global_step={step})")
    return agent


# Single-condition rollout
@torch.no_grad()
def evaluate_condition(
    agent: Agent,
    fixed_params: dict | None,
    n_episodes: int,
    max_episode_steps: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    env = AcrobotBalanceEnv(
        randomize=False,
        fixed_params=fixed_params or {},
        max_episode_steps=max_episode_steps,
    )
    returns, upright_fracs, steps_upright = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return, ep_up, ep_total = 0.0, 0, 0
        done = False
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs_t)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            ep_return += reward
            ep_up     += int(info.get("is_upright", False))
            ep_total  += 1
            done = terminated or truncated

        frac = ep_up / max(ep_total, 1)
        returns.append(ep_return)
        upright_fracs.append(frac)
        steps_upright.append(ep_up)

    env.close()
    return {
        "mean_return":          float(np.mean(returns)),
        "std_return":           float(np.std(returns)),
        "upright_fraction":     float(np.mean(upright_fracs)),
        "std_upright_fraction": float(np.std(upright_fracs)),
        "success_rate":         float(np.mean([f >= SUCCESS_THRESH for f in upright_fracs])),
        "steps_upright_mean":   float(np.mean(steps_upright)),
    }


# Full sweep
def run_evaluation(
    checkpoint_path: str,
    n_episodes: int = 100,
    max_episode_steps: int = 500,
    seed: int = 0,
    n_random_seeds: int = 5,
) -> pd.DataFrame:
    device = torch.device("cpu")
    agent  = load_agent(checkpoint_path, device)

    rows = []
    t0   = time.time()

    for mismatch in MISMATCH_LEVELS:
        for direction in DIRECTIONS:
            if mismatch == 0.0 and direction != "positive":
                continue  # nominal is direction-agnostic

            label = (f"±{int(mismatch*100):2d}% [{direction:8s}]"
                     if mismatch > 0.0 else " nominal")

            if direction == "random" and mismatch > 0.0:
                # Random mismatch can be noisy, so average a few parameter draws.
                all_m = []
                for rs in range(n_random_seeds):
                    np.random.seed(seed + rs * 999)
                    params = build_eval_params(mismatch, "random")
                    all_m.append(evaluate_condition(
                        agent, params, n_episodes, max_episode_steps, device, seed))
                metrics = {k: float(np.mean([d[k] for d in all_m])) for k in all_m[0]}
            else:
                params = (build_eval_params(mismatch, direction)
                          if mismatch > 0.0 else None)
                metrics = evaluate_condition(
                    agent, params, n_episodes, max_episode_steps, device, seed)

            row = {
                "mismatch_pct": int(mismatch * 100),
                "direction":    direction if mismatch > 0.0 else "nominal",
                **metrics,
            }
            rows.append(row)
            print(f"  {label}  "
                  f"return={metrics['mean_return']:6.2f}  "
                  f"upright={metrics['upright_fraction']:.3f}  "
                  f"success={metrics['success_rate']:.3f}")

    print(f"\nEvaluation complete in {time.time()-t0:.1f}s")
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/eval_results.csv", index=False)
    print("Results saved → results/eval_results.csv")
    return df


# Pretty table
def print_summary(df: pd.DataFrame):
    print("\n" + "═"*85)
    print("EVALUATION SUMMARY")
    print("═"*85)
    print(f"{'Mismatch':>10}  {'Direction':>10}  {'Return':>10}  "
          f"{'Upright%':>10}  {'Success%':>10}  {'Steps↑':>10}")
    print("─"*85)
    for _, row in df.iterrows():
        ms = f"±{row['mismatch_pct']}%" if row['mismatch_pct'] > 0 else "nominal"
        print(f"{ms:>10}  {row['direction']:>10}  "
              f"{row['mean_return']:>10.2f}  "
              f"{row['upright_fraction']*100:>9.1f}%  "
              f"{row['success_rate']*100:>9.1f}%  "
              f"{row['steps_upright_mean']:>10.1f}")
    print("═"*85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--n-episodes",        type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed",              type=int, default=0)
    parser.add_argument("--n-random-seeds",    type=int, default=5)
    args = parser.parse_args()

    df = run_evaluation(
        checkpoint_path  = args.checkpoint,
        n_episodes       = args.n_episodes,
        max_episode_steps= args.max_episode_steps,
        seed             = args.seed,
        n_random_seeds   = args.n_random_seeds,
    )
    print_summary(df)
