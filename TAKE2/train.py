"""
train.py  –  Robust Acrobot via PPO + Domain Randomization
===========================================================
Algorithm : PPO (Proximal Policy Optimization)
            Based on CleanRL: https://github.com/vwxyzjn/cleanrl
Robustness: Domain Randomization – physical parameters are sampled
            uniformly from [nominal*(1-dr_range), nominal*(1+dr_range)]
            at every episode reset during training.

Usage
-----
# Quick training run (CPU, ~5 min)
python train.py

# Custom settings
python train.py --total-timesteps 2000000 --dr-range 0.15 --num-envs 8

# Disable domain randomization (baseline)
python train.py --dr-range 0.0 --exp-name ppo_no_dr
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

import sys
sys.path.insert(0, os.path.dirname(__file__))
from envs.randomized_acrobot import make_env, NOMINAL_PARAMS


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a robust Acrobot policy with PPO + Domain Randomization"
    )
    # ---- Experiment ----
    parser.add_argument("--exp-name",       type=str,   default="ppo_robust_acrobot",
                        help="Experiment name (used for checkpoint/log dirs)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--cuda",           action="store_true", default=False,
                        help="Use GPU if available")
    parser.add_argument("--track",          action="store_true", default=False,
                        help="Enable TensorBoard logging")

    # ---- Environment ----
    parser.add_argument("--dr-range",       type=float, default=0.10,
                        help="Domain randomization range (0 = no DR, 0.1 = ±10%%)")
    parser.add_argument("--num-envs",       type=int,   default=4,
                        help="Number of parallel environments")
    parser.add_argument("--balance-reset-prob", type=float, default=0.20,
                        help="Training-only probability of starting near upright for balance curriculum")

    # ---- PPO Hyperparameters ----
    parser.add_argument("--total-timesteps",type=int,   default=2_000_000)
    parser.add_argument("--learning-rate",  type=float, default=2.5e-4)
    parser.add_argument("--num-steps",      type=int,   default=128,
                        help="Steps per rollout per environment")
    parser.add_argument("--anneal-lr",      action="store_true", default=True)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--gae-lambda",     type=float, default=0.95)
    parser.add_argument("--num-minibatches",type=int,   default=4)
    parser.add_argument("--update-epochs",  type=int,   default=10)
    parser.add_argument("--clip-coef",      type=float, default=0.2)
    parser.add_argument("--ent-coef",       type=float, default=0.01)
    parser.add_argument("--vf-coef",        type=float, default=0.5)
    parser.add_argument("--max-grad-norm",  type=float, default=0.5)

    # ---- Output ----
    parser.add_argument("--save-dir",       type=str,   default="checkpoints",
                        help="Directory to save model checkpoints")

    args = parser.parse_args()
    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


# ============================================================
# Neural Network (Actor-Critic)
# ============================================================

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    """Orthogonal initialization as recommended by CleanRL & PPO paper."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic network for discrete action spaces.

    Architecture
    ------------
    Input  (6,) → [128 → Tanh → 128 → Tanh]  ← shared representation
                                               ↓          ↓
                                         actor head   critic head
                                          (3 logits)   (1 value)

    We use a slightly larger hidden size (128 vs CleanRL's 64) to give the
    policy enough capacity to handle varied dynamics from domain randomization.
    """

    def __init__(self, obs_dim: int = 6, action_dim: int = 3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.actor_head  = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(128, 1),          std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.shared(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        hidden = self.shared(x)
        logits = self.actor_head(hidden)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic_head(hidden)

    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.shared(x)
        logits = self.actor_head(hidden)
        return torch.argmax(logits, dim=-1)


# ============================================================
# Training loop
# ============================================================

def train(args):
    run_name = f"{args.exp_name}_dr{args.dr_range}_seed{args.seed}_{int(time.time())}"
    print(f"\n{'='*60}")
    print(f"  Robust Acrobot PPO Training")
    print(f"{'='*60}")
    print(f"  Run          : {run_name}")
    print(f"  DR range     : ±{args.dr_range*100:.0f}%")
    print(f"  Balance reset: {args.balance_reset_prob*100:.0f}%")
    print(f"  Total steps  : {args.total_timesteps:,}")
    print(f"  Num envs     : {args.num_envs}")
    print(f"  Batch size   : {args.batch_size:,}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # TensorBoard (optional)
    # ------------------------------------------------------------------
    writer = None
    if args.track and _HAS_TB:
        log_dir = os.path.join("runs", run_name)
        writer  = SummaryWriter(log_dir)
        print(f"  TensorBoard: tensorboard --logdir {log_dir}")

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    envs = gym.vector.SyncVectorEnv(
        [make_env(seed=args.seed, idx=i, dr_range=args.dr_range,
                  balance_reset_prob=args.balance_reset_prob)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "This script requires a Discrete action space."

    obs_dim    = envs.single_observation_space.shape[0]  # 6
    action_dim = int(envs.single_action_space.n)          # 3

    # ------------------------------------------------------------------
    # Agent & Optimizer
    # ------------------------------------------------------------------
    agent     = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"  Network parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print()

    # ------------------------------------------------------------------
    # Rollout buffers
    # ------------------------------------------------------------------
    obs_buf     = torch.zeros(args.num_steps, args.num_envs, obs_dim).to(device)
    actions_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    logprobs_buf= torch.zeros(args.num_steps, args.num_envs).to(device)
    rewards_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    dones_buf   = torch.zeros(args.num_steps, args.num_envs).to(device)
    values_buf  = torch.zeros(args.num_steps, args.num_envs).to(device)

    # ------------------------------------------------------------------
    # Initialise env
    # ------------------------------------------------------------------
    global_step  = 0
    start_time   = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs     = torch.Tensor(next_obs_np).to(device)
    next_done    = torch.zeros(args.num_envs).to(device)

    num_updates  = args.total_timesteps // args.batch_size
    episode_returns: list[float] = []
    episode_lengths: list[int]   = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for update in range(1, num_updates + 1):

        # ---- Learning rate annealing ----
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ================================================================
        # Phase 1: Collect rollout
        # ================================================================
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]  = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step]  = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = terminated | truncated
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs  = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done.astype(np.float32)).to(device)

            # Collect completed episode stats
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_ret = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        episode_returns.append(float(ep_ret))
                        episode_lengths.append(int(ep_len))

        # ================================================================
        # Phase 2: Compute GAE advantages
        # ================================================================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            last_gae   = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_val = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    next_val = values_buf[t + 1]
                delta   = rewards_buf[t] + args.gamma * next_val * nextnonterminal - values_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        # ================================================================
        # Phase 3: PPO update
        # ================================================================
        b_obs        = obs_buf.reshape((-1, obs_dim))
        b_actions    = actions_buf.reshape(-1)
        b_logprobs   = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values_buf.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs: list[float] = []

        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end  = start + args.minibatch_size
                mb   = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb].long()
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio    = logratio.exp()

                # Diagnose policy drift
                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Normalise advantages within minibatch
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss (clipped PPO)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                nv = newvalue.view(-1)
                v_loss_unclipped = (nv - b_returns[mb]) ** 2
                v_clipped        = b_values[mb] + torch.clamp(nv - b_values[mb], -args.clip_coef, args.clip_coef)
                v_loss_clipped   = (v_clipped - b_returns[mb]) ** 2
                v_loss           = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ================================================================
        # Logging
        # ================================================================
        if update % 10 == 0 or update == 1:
            sps = int(global_step / (time.time() - start_time))
            if len(episode_returns) > 0:
                window  = episode_returns[-50:]
                mean_r  = np.mean(window)
                success = sum(1 for l in episode_lengths[-50:] if l < 500) / min(50, len(episode_lengths[-50:]))
                print(
                    f"  Update {update:5d}/{num_updates} | "
                    f"Step {global_step:8,} | "
                    f"Return {mean_r:7.1f} | "
                    f"Success {success*100:5.1f}% | "
                    f"SPS {sps:6,}"
                )
                if writer:
                    writer.add_scalar("charts/mean_episodic_return", mean_r, global_step)
                    writer.add_scalar("charts/success_rate", success, global_step)
            else:
                print(f"  Update {update:5d}/{num_updates} | Step {global_step:8,} | "
                      f"(waiting for first episode...) | SPS {sps:6,}")

            if writer:
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/value_loss",  v_loss.item(),  global_step)
                writer.add_scalar("losses/entropy",     entropy_loss.item(), global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

    # ================================================================
    # Save checkpoint
    # ================================================================
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{run_name}.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "args": vars(args),
        "obs_dim":    obs_dim,
        "action_dim": action_dim,
        "global_step": global_step,
        "run_name":   run_name,
    }, ckpt_path)

    # Also save a "latest" symlink for easy evaluation
    latest_path = os.path.join(args.save_dir, "latest.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "args": vars(args),
        "obs_dim":    obs_dim,
        "action_dim": action_dim,
        "global_step": global_step,
        "run_name":   run_name,
    }, latest_path)

    total_time = time.time() - start_time
    final_window = episode_returns[-100:] if len(episode_returns) >= 100 else episode_returns
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total wall time : {total_time/60:.1f} min")
    print(f"  Final return    : {np.mean(final_window):.1f} ± {np.std(final_window):.1f}")
    print(f"  Model saved     : {ckpt_path}")
    print(f"{'='*60}\n")

    if writer:
        writer.close()

    envs.close()
    return ckpt_path


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
