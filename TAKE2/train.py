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
import torch.nn.functional as F
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
from envs.randomized_acrobot import MAX_EPISODE_STEPS, make_env


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
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Optional checkpoint to initialize the policy before training")

    # ---- Environment ----
    parser.add_argument("--dr-range",       type=float, default=0.05,
                        help="Domain randomization range (0 = no DR, 0.05 = ±5%%)")
    parser.add_argument("--num-envs",       type=int,   default=16,
                        help="Number of parallel environments")
    parser.add_argument("--balance-reset-prob", type=float, default=0.90,
                        help="Training-only probability of starting near upright for balance curriculum")

    # ---- PPO Hyperparameters ----
    parser.add_argument("--total-timesteps",type=int,   default=10_000_000)
    parser.add_argument("--learning-rate",  type=float, default=2.5e-4)
    parser.add_argument("--num-steps",      type=int,   default=256,
                        help="Steps per rollout per environment")
    parser.add_argument("--anneal-lr",      action="store_true", default=True)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--gae-lambda",     type=float, default=0.95)
    parser.add_argument("--num-minibatches",type=int,   default=8)
    parser.add_argument("--update-epochs",  type=int,   default=10)
    parser.add_argument("--clip-coef",      type=float, default=0.2)
    parser.add_argument("--ent-coef",       type=float, default=0.01)
    parser.add_argument("--vf-coef",        type=float, default=0.5)
    parser.add_argument("--max-grad-norm",  type=float, default=0.5)
    parser.add_argument("--bc-coef",        type=float, default=1.00,
                        help="Initial auxiliary behavior-cloning weight for MPC actions in capture states")
    parser.add_argument("--bc-final-coef",  type=float, default=0.25,
                        help="Final auxiliary behavior-cloning weight after linear annealing")
    parser.add_argument("--teacher-action-prob", type=float, default=0.60,
                        help="Initial probability of executing the teacher action in capture states")
    parser.add_argument("--teacher-final-prob", type=float, default=0.0,
                        help="Final teacher action probability after linear annealing")
    parser.add_argument("--pretrain-bc-steps", type=int, default=0,
                        help="Supervised teacher-imitation updates before PPO")
    parser.add_argument("--pretrain-bc-batch-size", type=int, default=1024,
                        help="Batch size for supervised BC pretraining")
    parser.add_argument("--pretrain-bc-lr", type=float, default=1e-3,
                        help="Learning rate for supervised BC pretraining")
    parser.add_argument("--pretrain-eval-interval", type=int, default=500,
                        help="BC pretraining updates between policy-only hold checks")
    parser.add_argument("--pretrain-reset-fraction", type=float, default=0.75,
                        help="Fraction of BC samples drawn from fresh near-upright resets")
    parser.add_argument("--pretrain-policy-fraction", type=float, default=0.50,
                        help="Fraction of BC collection steps executed by the current policy")
    parser.add_argument("--pretrain-reset-on-fall", action=argparse.BooleanOptionalAction, default=True,
                        help="Reset BC collection when a rollout leaves the capture region")

    # ---- Output ----
    parser.add_argument("--save-dir",       type=str,   default="checkpoints",
                        help="Directory to save model checkpoints")

    args = parser.parse_args()
    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


def vector_env_call(envs: gym.vector.SyncVectorEnv, name: str) -> np.ndarray:
    """Call an unwrapped environment method across SyncVectorEnv workers."""
    try:
        return np.asarray(envs.call(name))
    except Exception:
        return np.asarray([getattr(env.unwrapped, name)() for env in envs.envs])


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
    Input  (10,) -> [256 -> Tanh -> 256 -> Tanh] <- shared representation
                                               ↓          ↓
                                         actor head   critic head
                                          (11 logits)  (1 value)

    We use a larger hidden size (256 vs CleanRL's 64) to give the
    policy enough capacity to handle varied dynamics from domain randomization.
    """

    def __init__(self, obs_dim: int = 10, action_dim: int = 11) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.actor_head  = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(256, 1),          std=1.0)

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


def collect_teacher_batch(
    agent: ActorCritic,
    env: gym.Env,
    batch_size: int,
    device: torch.device,
    reset_fraction: float,
    policy_fraction: float,
    reset_on_fall: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_batch = np.zeros((batch_size, env.observation_space.shape[0]), dtype=np.float32)
    action_batch = np.zeros(batch_size, dtype=np.int64)
    obs, _ = env.reset()

    for idx in range(batch_size):
        if idx == 0 or np.random.random() < reset_fraction:
            obs, _ = env.reset()
        teacher_action = int(env.unwrapped.expert_action())
        obs_batch[idx] = obs
        action_batch[idx] = teacher_action
        if np.random.random() < policy_fraction:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.get_deterministic_action(obs_tensor).item())
        else:
            action = teacher_action
        obs, _, terminated, truncated, _ = env.step(action)
        if (
            terminated
            or truncated
            or env.unwrapped._episode_steps >= MAX_EPISODE_STEPS
            or (reset_on_fall and not env.unwrapped.should_use_balance_controller())
        ):
            obs, _ = env.reset()

    return (
        torch.as_tensor(obs_batch, dtype=torch.float32, device=device),
        torch.as_tensor(action_batch, dtype=torch.long, device=device),
    )


def evaluate_hold_policy(
    agent: ActorCritic,
    episodes: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    env = make_env(
        seed=seed,
        idx=0,
        dr_range=0.0,
        balance_reset_prob=1.0,
    )()
    holds: list[int] = []
    balance_times: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        max_hold = 0
        balanced_steps = 0
        steps = 0
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.get_deterministic_action(obs_tensor).item())
            obs, _, terminated, truncated, info = env.step(action)
            max_hold = max(max_hold, int(info.get("episode_max_balance_steps", 0)))
            balanced_steps += int(info.get("balanced", False))
            steps += 1
            done = terminated or truncated
        holds.append(max_hold)
        balance_times.append(balanced_steps / max(1, steps))

    env.close()
    return float(np.mean(holds)), float(np.mean(balance_times))


def pretrain_behavior_cloning(
    agent: ActorCritic,
    args,
    device: torch.device,
) -> None:
    if args.pretrain_bc_steps <= 0:
        return

    env = make_env(
        seed=args.seed + 10_000,
        idx=0,
        dr_range=args.dr_range,
        balance_reset_prob=1.0,
    )()
    optimizer = optim.Adam(agent.parameters(), lr=args.pretrain_bc_lr, eps=1e-5)

    print("\n" + "=" * 60)
    print("  Behavior Cloning Pretraining")
    print("=" * 60)
    print(f"  Updates     : {args.pretrain_bc_steps:,}")
    print(f"  Batch size  : {args.pretrain_bc_batch_size:,}")
    print(f"  Reset frac  : {args.pretrain_reset_fraction:.2f}")
    print(f"  Policy frac : {args.pretrain_policy_fraction:.2f}")
    print(f"  Reset fall  : {args.pretrain_reset_on_fall}")
    print(f"  LR          : {args.pretrain_bc_lr:g}")
    print("=" * 60)

    for update in range(1, args.pretrain_bc_steps + 1):
        obs_batch, action_batch = collect_teacher_batch(
            agent,
            env,
            args.pretrain_bc_batch_size,
            device,
            args.pretrain_reset_fraction,
            args.pretrain_policy_fraction,
            args.pretrain_reset_on_fall,
        )
        hidden = agent.shared(obs_batch)
        logits = agent.actor_head(hidden)
        loss = F.cross_entropy(logits, action_batch)
        with torch.no_grad():
            acc = (torch.argmax(logits, dim=-1) == action_batch).float().mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        if update == 1 or update % args.pretrain_eval_interval == 0:
            mean_hold, mean_bal = evaluate_hold_policy(
                agent=agent,
                episodes=10,
                seed=args.seed + 20_000,
                device=device,
            )
            print(
                f"  BC update {update:5d}/{args.pretrain_bc_steps} | "
                f"loss {loss.item():6.3f} | acc {acc.item()*100:5.1f}% | "
                f"Hold {mean_hold:6.1f} | BalTime {mean_bal*100:5.1f}%"
            )

    env.close()


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
    print(f"  BC coef      : {args.bc_coef:.3f} -> {args.bc_final_coef:.3f}")
    print(f"  Teacher act  : {args.teacher_action_prob:.3f} -> {args.teacher_final_prob:.3f}")
    print(f"  BC pretrain  : {args.pretrain_bc_steps:,} updates")
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

    obs_dim    = envs.single_observation_space.shape[0]  # 10
    action_dim = int(envs.single_action_space.n)          # 11 torque levels

    # ------------------------------------------------------------------
    # Agent & Optimizer
    # ------------------------------------------------------------------
    agent     = ActorCritic(obs_dim, action_dim).to(device)
    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        ckpt_obs_dim = ckpt.get("obs_dim", obs_dim)
        ckpt_action_dim = ckpt.get("action_dim", action_dim)
        if ckpt_obs_dim != obs_dim or ckpt_action_dim != action_dim:
            raise ValueError(
                f"Checkpoint shape mismatch: checkpoint obs/action "
                f"{ckpt_obs_dim}/{ckpt_action_dim}, current {obs_dim}/{action_dim}"
            )
        agent.load_state_dict(ckpt["model_state_dict"])
        print(f"  Resumed from: {args.resume_checkpoint}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"  Network parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print()

    pretrain_behavior_cloning(agent, args, device)

    # ------------------------------------------------------------------
    # Rollout buffers
    # ------------------------------------------------------------------
    obs_buf     = torch.zeros(args.num_steps, args.num_envs, obs_dim).to(device)
    actions_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    logprobs_buf= torch.zeros(args.num_steps, args.num_envs).to(device)
    rewards_buf = torch.zeros(args.num_steps, args.num_envs).to(device)
    dones_buf   = torch.zeros(args.num_steps, args.num_envs).to(device)
    values_buf  = torch.zeros(args.num_steps, args.num_envs).to(device)
    teacher_actions_buf = torch.zeros(args.num_steps, args.num_envs, dtype=torch.long).to(device)
    teacher_masks_buf   = torch.zeros(args.num_steps, args.num_envs).to(device)

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
    episode_max_holds: list[int] = []
    episode_upright_time: list[float] = []
    episode_balance_time: list[float] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for update in range(1, num_updates + 1):

        # ---- Learning rate annealing ----
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        schedule_frac = 1.0 - (update - 1.0) / max(1, num_updates - 1)
        teacher_action_prob = args.teacher_final_prob + (
            args.teacher_action_prob - args.teacher_final_prob
        ) * schedule_frac
        guide_fracs: list[float] = []

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
                if args.bc_coef > 0.0 or teacher_action_prob > 0.0:
                    teacher_mask_np = vector_env_call(envs, "should_use_balance_controller").astype(bool)
                    teacher_actions_np = np.zeros(args.num_envs, dtype=np.int64)
                    if np.any(teacher_mask_np):
                        teacher_actions_np = vector_env_call(envs, "expert_action").astype(np.int64)
                    teacher_actions = torch.as_tensor(
                        teacher_actions_np,
                        dtype=torch.long,
                        device=device,
                    )
                    teacher_actions_buf[step] = teacher_actions
                    teacher_masks_buf[step] = torch.as_tensor(
                        teacher_mask_np.astype(np.float32),
                        dtype=torch.float32,
                        device=device,
                    )
                    if teacher_action_prob > 0.0 and np.any(teacher_mask_np):
                        guide_mask_np = (
                            teacher_mask_np
                            & (np.random.random(args.num_envs) < teacher_action_prob)
                        )
                        guide_fracs.append(float(np.mean(guide_mask_np)))
                        if np.any(guide_mask_np):
                            guide_mask = torch.as_tensor(
                                guide_mask_np,
                                dtype=torch.bool,
                                device=device,
                            )
                            action = action.clone()
                            action[guide_mask] = teacher_actions[guide_mask]
                            _, logprob, _, value = agent.get_action_and_value(next_obs, action.long())
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
                        episode_max_holds.append(
                            int(info.get("episode_max_balance_steps", info.get("balance_steps", 0)))
                        )
                        episode_upright_time.append(float(info.get("upright_time_fraction", 0.0)))
                        episode_balance_time.append(float(info.get("balance_time_fraction", 0.0)))
            elif "episode" in infos:
                done_mask = infos.get("_episode", np.zeros(args.num_envs, dtype=bool))
                for env_idx, episode_done in enumerate(done_mask):
                    if not episode_done:
                        continue
                    ep_ret = infos["episode"]["r"][env_idx]
                    ep_len = infos["episode"]["l"][env_idx]
                    episode_returns.append(float(ep_ret))
                    episode_lengths.append(int(ep_len))
                    episode_max_holds.append(
                        int(infos.get("episode_max_balance_steps", np.zeros(args.num_envs))[env_idx])
                    )
                    episode_upright_time.append(
                        float(infos.get("upright_time_fraction", np.zeros(args.num_envs))[env_idx])
                    )
                    episode_balance_time.append(
                        float(infos.get("balance_time_fraction", np.zeros(args.num_envs))[env_idx])
                    )

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
        b_teacher_actions = teacher_actions_buf.reshape(-1)
        b_teacher_masks   = teacher_masks_buf.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs: list[float] = []
        bc_losses: list[float] = []
        bc_accs: list[float] = []
        bc_weight = args.bc_final_coef + (
            args.bc_coef - args.bc_final_coef
        ) * (1.0 - (update - 1.0) / max(1, num_updates - 1))

        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end  = start + args.minibatch_size
                mb   = b_inds[start:end]

                hidden = agent.shared(b_obs[mb])
                logits = agent.actor_head(hidden)
                dist = Categorical(logits=logits)
                newlogprob = dist.log_prob(b_actions[mb].long())
                entropy = dist.entropy()
                newvalue = agent.critic_head(hidden)
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

                # Auxiliary balance teacher: imitate the MPC stabilizer only
                # on high/near-upright capture states collected in the rollout.
                teacher_mask = b_teacher_masks[mb] > 0.5
                if args.bc_coef > 0.0 and torch.any(teacher_mask):
                    bc_loss = F.cross_entropy(
                        logits[teacher_mask],
                        b_teacher_actions[mb][teacher_mask].long(),
                    )
                    with torch.no_grad():
                        bc_acc = (
                            torch.argmax(logits[teacher_mask], dim=-1)
                            == b_teacher_actions[mb][teacher_mask].long()
                        ).float().mean()
                    bc_losses.append(float(bc_loss.item()))
                    bc_accs.append(float(bc_acc.item()))
                else:
                    bc_loss = torch.zeros((), device=device)

                # Total loss
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + bc_weight * bc_loss
                )

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
                mean_hold = np.mean(episode_max_holds[-50:]) if episode_max_holds else 0.0
                upright = np.mean(episode_upright_time[-50:]) if episode_upright_time else 0.0
                balanced = np.mean(episode_balance_time[-50:]) if episode_balance_time else 0.0
                print(
                    f"  Update {update:5d}/{num_updates} | "
                    f"Step {global_step:8,} | "
                    f"Return {mean_r:7.1f} | "
                    f"HoldMax {mean_hold:5.1f} | "
                    f"Upright {upright*100:5.1f}% | "
                    f"BalTime {balanced*100:5.1f}% | "
                    f"BC {np.mean(bc_losses) if bc_losses else 0.0:5.2f}/"
                    f"{np.mean(bc_accs)*100 if bc_accs else 0.0:4.0f}% | "
                    f"Guide {np.mean(guide_fracs)*100 if guide_fracs else 0.0:4.0f}% | "
                    f"SPS {sps:6,}"
                )
                if writer:
                    writer.add_scalar("charts/mean_episodic_return", mean_r, global_step)
                    writer.add_scalar("charts/mean_max_hold_steps", mean_hold, global_step)
                    writer.add_scalar("charts/hold_score_fraction", mean_hold / MAX_EPISODE_STEPS, global_step)
                    writer.add_scalar("charts/upright_time_fraction", upright, global_step)
                    writer.add_scalar("charts/balance_time_fraction", balanced, global_step)
                    writer.add_scalar("losses/bc_loss", np.mean(bc_losses) if bc_losses else 0.0, global_step)
                    writer.add_scalar("charts/bc_accuracy", np.mean(bc_accs) if bc_accs else 0.0, global_step)
                    writer.add_scalar("charts/bc_weight", bc_weight, global_step)
                    writer.add_scalar("charts/teacher_action_prob", teacher_action_prob, global_step)
                    writer.add_scalar("charts/teacher_action_fraction", np.mean(guide_fracs) if guide_fracs else 0.0, global_step)
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
    final_holds = episode_max_holds[-100:] if len(episode_max_holds) >= 100 else episode_max_holds
    final_upright = episode_upright_time[-100:] if len(episode_upright_time) >= 100 else episode_upright_time
    final_balance = episode_balance_time[-100:] if len(episode_balance_time) >= 100 else episode_balance_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total wall time : {total_time/60:.1f} min")
    if final_window:
        print(f"  Final return    : {np.mean(final_window):.1f} +/- {np.std(final_window):.1f}")
        print(f"  Final HoldMax   : {np.mean(final_holds):.1f} / {MAX_EPISODE_STEPS}")
        print(f"  Final Upright   : {np.mean(final_upright)*100:.1f}%")
        print(f"  Final BalTime   : {np.mean(final_balance)*100:.1f}%")
    else:
        print("  Final metrics   : no completed episodes")
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
