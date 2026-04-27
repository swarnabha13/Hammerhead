"""
train.py — PPO + Domain Randomisation for Acrobot Swing-Up AND Balance
=======================================================================
Based on CleanRL's ppo.py (https://github.com/vwxyzjn/cleanrl).

Goal: Train a policy that
  1. Swings the Acrobot up so the free tip clears the horizontal line, AND
  2. Holds BOTH links upright for at least 100 consecutive steps.

Key differences from swing-up-only version
  - AcrobotBalanceEnv    : dense shaped reward + new termination condition
  - max_episode_steps=1000: double the standard horizon (swing takes time)
  - Network 128×128      : larger to handle the harder policy
  - total_timesteps=2M   : balance is a harder credit-assignment problem
  - gamma=0.995          : higher discount to value sustained balance
  - ent_coef=0.02        : slightly more exploration entropy

Usage
-----
    python train.py                          # defaults: 2M steps, ±20% DR
    python train.py --total-timesteps 3000000
    python train.py --cuda --rand-range 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from common import Agent, make_balance_env, UPRIGHT_HEIGHT, BALANCE_STEPS_WIN


@dataclass
class TrainConfig:
    seed: int            = 42
    cuda: bool           = False

    # Domain randomisation
    rand_range: float    = 0.20       # ±20% physics perturbation

    # Episode / horizon
    max_episode_steps: int = 1000     # 2× standard (swing takes time)

    # PPO / training schedule
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int        = 8          # more parallel envs for balance diversity
    num_steps: int       = 256        # longer rollout to see balance episodes
    anneal_lr: bool      = True
    gamma: float         = 0.995      # high discount for long-horizon balance
    gae_lambda: float    = 0.95
    num_minibatches: int = 8
    update_epochs: int   = 4
    norm_adv: bool       = True
    clip_coef: float     = 0.2
    clip_vloss: bool     = True
    ent_coef: float      = 0.02       # slightly more exploration
    vf_coef: float       = 0.5
    max_grad_norm: float = 0.5

    save_dir: str        = "./results"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def train(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
    )

    print(f"\n{'='*60}")
    print(f"  Acrobot Swing-Up + Balance — PPO + Domain Randomisation")
    print(f"{'='*60}")
    print(f"  Device          : {device}")
    print(f"  Rand range      : ±{cfg.rand_range*100:.0f}%  (7 physics params)")
    print(f"  Total steps     : {cfg.total_timesteps:,}")
    print(f"  Episode horizon : {cfg.max_episode_steps} steps")
    print(f"  Balance target  : {BALANCE_STEPS_WIN} consecutive upright steps")
    print(f"  Upright thresh  : tip_height ≥ {UPRIGHT_HEIGHT}")
    print(f"  Batch size      : {cfg.batch_size}  "
          f"({cfg.num_envs} envs × {cfg.num_steps} steps)")
    print(f"  Save dir        : {cfg.save_dir}")
    print(f"{'='*60}\n")

    envs = gym.vector.SyncVectorEnv([
        make_balance_env(cfg.seed, i,
                         randomize=True,
                         rand_range=cfg.rand_range,
                         max_episode_steps=cfg.max_episode_steps)
        for i in range(cfg.num_envs)
    ])

    obs_dim   = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)

    agent     = Agent(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs_shape = envs.single_observation_space.shape
    obs      = torch.zeros((cfg.num_steps, cfg.num_envs) + obs_shape).to(device)
    actions  = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards  = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones    = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values   = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    global_step = 0
    start_time  = time.time()
    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)

    num_updates          = cfg.total_timesteps // cfg.batch_size
    ep_returns: List[float] = []
    ep_lengths: List[int]   = []
    ep_successes: List[int] = []   # 1 = held balance for 100 steps

    for update in range(1, num_updates + 1):

        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ── Rollout ────────────────────────────────────────────────────────────
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs[step]   = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step]  = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(
                reward, dtype=torch.float32, device=device)
            next_obs  = torch.tensor(
                next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(
                done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_returns.append(float(info["episode"]["r"]))
                        ep_lengths.append(int(info["episode"]["l"]))
                        # Check if this episode achieved sustained balance
                        ep_successes.append(
                            int(info.get("upright_count", 0) >= BALANCE_STEPS_WIN)
                        )

        # ── GAE ────────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues      = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues      = values[t + 1]
                delta         = (rewards[t]
                                 + cfg.gamma * nextvalues * nextnonterminal
                                 - values[t])
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda
                    * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # ── Flatten ────────────────────────────────────────────────────────────
        b_obs      = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions  = actions.reshape(-1).long()
        b_adv      = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)
        b_values   = values.reshape(-1)

        # ── PPO update ─────────────────────────────────────────────────────────
        b_inds = np.arange(cfg.batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb  = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio    = logratio.exp()

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(
                        ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_clip = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb],
                        -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max(
                        (newvalue - b_returns[mb]) ** 2,
                        (v_clip   - b_returns[mb]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                loss = (pg_loss
                        - cfg.ent_coef * entropy.mean()
                        + cfg.vf_coef  * v_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # ── Logging ────────────────────────────────────────────────────────────
        if update % 20 == 0 or update == num_updates:
            sps        = int(global_step / (time.time() - start_time + 1e-9))
            w_ret      = ep_returns[-100:]   if ep_returns   else [0.0]
            w_succ     = ep_successes[-100:] if ep_successes else [0]
            mean_ret   = np.mean(w_ret)
            succ_rate  = np.mean(w_succ) * 100
            print(
                f"  update {update:4d}/{num_updates} | "
                f"steps {global_step:8,d} | "
                f"mean_return {mean_ret:8.2f} | "
                f"balance_success {succ_rate:5.1f}% | "
                f"{sps:5d} SPS"
            )

    # ── Save ───────────────────────────────────────────────────────────────────
    model_path   = os.path.join(cfg.save_dir, "ppo_acrobot.pth")
    history_path = os.path.join(cfg.save_dir, "training_history.json")

    torch.save({
        "agent_state_dict": agent.state_dict(),
        "obs_dim":    obs_dim,
        "n_actions":  n_actions,
        "config":     asdict(cfg),
    }, model_path)

    with open(history_path, "w") as fh:
        json.dump({
            "episode_returns":   ep_returns,
            "episode_lengths":   ep_lengths,
            "episode_successes": ep_successes,
            "config":            asdict(cfg),
        }, fh, indent=2)

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Model   → {model_path}")
    print(f"  History → {history_path}\n")

    envs.close()
    return agent, ep_returns, ep_successes


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train Acrobot swing-up + balance policy with PPO + DR")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--cuda",              action="store_true")
    p.add_argument("--rand-range",        type=float, default=0.20)
    p.add_argument("--total-timesteps",   type=int,   default=2_000_000)
    p.add_argument("--max-episode-steps", type=int,   default=1000)
    p.add_argument("--learning-rate",     type=float, default=2.5e-4)
    p.add_argument("--num-envs",          type=int,   default=8)
    p.add_argument("--num-steps",         type=int,   default=256)
    p.add_argument("--num-minibatches",   type=int,   default=8)
    p.add_argument("--update-epochs",     type=int,   default=4)
    p.add_argument("--ent-coef",          type=float, default=0.02)
    p.add_argument("--gamma",             type=float, default=0.995)
    p.add_argument("--save-dir",          type=str,   default="./results")
    a = p.parse_args()
    return TrainConfig(
        seed              = a.seed,
        cuda              = a.cuda,
        rand_range        = a.rand_range,
        total_timesteps   = a.total_timesteps,
        max_episode_steps = a.max_episode_steps,
        learning_rate     = a.learning_rate,
        num_envs          = a.num_envs,
        num_steps         = a.num_steps,
        num_minibatches   = a.num_minibatches,
        update_epochs     = a.update_epochs,
        ent_coef          = a.ent_coef,
        gamma             = a.gamma,
        save_dir          = a.save_dir,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
