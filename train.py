"""
train.py — PPO training for robust Acrobot balance.

Adapted from CleanRL (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
Modifications:
  1. Custom AcrobotBalanceEnv with dense reward and no early termination.
  2. Domain randomisation: physics params re-sampled every episode reset.
  3. Larger MLP (128 units) for expressivity across parameter distributions.
  4. Entropy coeff 0.02 to encourage early exploration.

Usage:
  python train.py
  python train.py --total-timesteps 3000000 --dr-range 0.2 --num-envs 16
"""

from __future__ import annotations
import os, random, time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from envs.acrobot_custom import make_env


@dataclass
class Args:
    exp_name: str = "ppo_acrobot_dr"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 2_000_000
    max_episode_steps: int = 500
    dr_range: float = 0.20
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    learning_rate: float = 3e-4
    hidden_size: int = 128


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train(args: Args) -> str:
    # Keep these derived from the CLI arguments so quick and full runs do not
    # accidentally use stale buffer sizes.
    batch_size     = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs("runs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[train] device={device}  run={run_name}")
    print(f"[train] batch={batch_size}  minibatch={minibatch_size}  iterations={num_iterations}")

    envs = gym.vector.SyncVectorEnv([
        make_env(randomize=True, dr_range=args.dr_range,
                 max_episode_steps=args.max_episode_steps,
                 seed=args.seed, idx=i, run_name=run_name)
        for i in range(args.num_envs)
    ])

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    agent = Agent(obs_dim, act_dim, hidden=args.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf      = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions_buf  = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf  = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf    = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf   = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs,  dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    best_mean_reward = -np.inf
    checkpoint_path = f"checkpoints/{run_name}_best.pt"

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Roll out the current policy before the PPO update.
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]   = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step]  = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None:
                        continue
                    ep_r  = info.get("episode", {}).get("r")
                    ep_l  = info.get("episode", {}).get("l")
                    uf    = info.get("upright_fraction", 0.0)
                    if ep_r is not None:
                        writer.add_scalar("charts/episodic_return",  ep_r, global_step)
                        writer.add_scalar("charts/episodic_length",  ep_l, global_step)
                        writer.add_scalar("charts/upright_fraction", uf,   global_step)

        # Standard GAE pass over the rollout buffer.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                nxt_nonterminal = 1.0 - (next_done if t == args.num_steps - 1 else dones_buf[t + 1])
                nxt_values      = next_value  if t == args.num_steps - 1 else values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nxt_values * nxt_nonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nxt_nonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs        = obs_buf.reshape((-1, obs_dim))
        b_actions    = actions_buf.reshape(-1)
        b_logprobs   = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = pg_loss = v_loss = entropy_loss = torch.tensor(0.0)

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds].long()
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        mean_reward = rewards_buf.mean().item()
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("losses/value_loss",       v_loss.item(),        global_step)
        writer.add_scalar("losses/policy_loss",      pg_loss.item(),       global_step)
        writer.add_scalar("losses/entropy",          entropy_loss.item(),  global_step)
        writer.add_scalar("losses/approx_kl",        approx_kl.item(),     global_step)
        writer.add_scalar("losses/clipfrac",         np.mean(clipfracs),   global_step)
        writer.add_scalar("charts/mean_step_reward", mean_reward,          global_step)
        writer.add_scalar("charts/SPS",              sps,                  global_step)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save({"agent_state_dict": agent.state_dict(),
                        "args": vars(args), "global_step": global_step},
                       checkpoint_path)

        if iteration % max(1, num_iterations // 10) == 0 or iteration == num_iterations:
            print(f"  [{iteration:4d}/{num_iterations}] step={global_step:>8,}  "
                  f"mean_rew={mean_reward:.4f}  v_loss={v_loss.item():.4f}  SPS={sps:,}")

    envs.close()
    writer.close()
    print(f"\nTraining complete. Best checkpoint → {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
