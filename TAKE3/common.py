"""
common.py — Shared components for Robust Acrobot Swing-Up + Balance
====================================================================
Key additions vs. the swing-up-only version:
  - AcrobotBalanceEnv  : custom wrapper that overrides reward & termination
                         so the agent must HOLD the inverted position, not
                         just cross the line once.
  - DomainRandomizedAcrobot: unchanged API; works with any inner env.
  - Agent              : same actor-critic MLP (64×64, tanh, orthogonal init)
  - make_balance_env   : factory for the full stack used in training

Acrobot coordinate convention (Gymnasium):
  θ₁ = angle of link 1 from the DOWNWARD vertical  (0 = hanging)
  θ₂ = relative angle between link 2 and link 1     (0 = straight)
  observation = [cos θ₁, sin θ₁, cos θ₂, sin θ₂, dθ₁, dθ₂]

  Fully upright:  θ₁ = π  →  cos θ₁ = -1
                  θ₂ = 0  →  cos θ₂ = +1
  Tip height     = -cos θ₁ - cos(θ₁+θ₂)
                 = -cos θ₁ - (cos θ₁ cos θ₂ - sin θ₁ sin θ₂)
  Range: [-2 (hanging), +2 (fully upright)]
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# Nominal Acrobot parameters
NOMINAL_PARAMS: Dict[str, float] = {
    "LINK_LENGTH_1":  1.0,
    "LINK_LENGTH_2":  1.0,
    "LINK_MASS_1":    1.0,
    "LINK_MASS_2":    1.0,
    "LINK_COM_POS_1": 0.5,
    "LINK_COM_POS_2": 0.5,
    "LINK_MOI":       1.0,
}

# Balance thresholds
UPRIGHT_HEIGHT     = 1.8   # tip height threshold for "upright" (max = 2.0)
BALANCE_STEPS_WIN  = 100   # consecutive upright steps needed for success


# Swing-up + balance wrapper
class AcrobotBalanceEnv(gym.Wrapper):
    """
    Wraps standard Acrobot-v1 and replaces reward / termination with a
    swing-up + balance objective.

    Reward sketch:
      r_height  = normalized tip height
      r_posture = weighted height/link-alignment score
      r_vel     = velocity penalty, weighted more near the top
      r_balance = per-step bonus once the tip is high enough
      success   = 100 consecutive upright steps

    Termination:
      - SUCCESS : upright for BALANCE_STEPS_WIN consecutive steps
      - TIMEOUT : max_episode_steps (handled by gym's TimeLimit wrapper)
      - The standard "tip crosses line" termination is SUPPRESSED.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._upright_count: int = 0

    @staticmethod
    def tip_height(obs: np.ndarray) -> float:
        c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
        return float(-c1 - (c1 * c2 - s1 * s2))

    def _shaped_reward(self, obs: np.ndarray) -> Tuple[float, bool]:
        c1, s1, c2, s2, dth1, dth2 = obs

        th  = self.tip_height(obs)
        r_h = (th + 2.0) / 4.0                         # height:  [0, 1]
        r1  = (-c1 + 1.0) / 2.0                        # link 1 up: [0, 1]
        r2  = ( c2 + 1.0) / 2.0                        # link 2 straight: [0, 1]

        posture     = 0.5 * r_h + 0.3 * r1 + 0.2 * r2
        near_top    = r_h ** 2                          # weight vel penalty near top
        vel_penalty = near_top * 0.05 * (dth1 ** 2 + dth2 ** 2)

        reward = posture - vel_penalty

        # Balance bonus and success detection.
        success = False
        if th >= UPRIGHT_HEIGHT:
            self._upright_count += 1
            reward += 0.5                               # per-step balance bonus
            if self._upright_count >= BALANCE_STEPS_WIN:
                reward  += 50.0                         # terminal success bonus
                success  = True
        else:
            self._upright_count = 0

        return reward, success

    def reset(self, **kwargs):
        self._upright_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Let the base env do the physics update.
        obs, _std_reward, _std_term, truncated, info = self.env.step(action)
        # Replace the base reward/termination with the balance objective.
        reward, success = self._shaped_reward(obs)
        info["upright_count"] = self._upright_count
        info["tip_height"]    = self.tip_height(obs)
        return obs, reward, success, truncated, info


# Domain randomization wrapper
class DomainRandomizedAcrobot(gym.Wrapper):
    """
    Re-samples physics parameters on every reset().

    Training (randomize=True):   each param ~ Uniform[nom*(1-r), nom*(1+r)]
    Evaluation (randomize=False): each param = nom * (1 + mismatch_level)
    """

    def __init__(
        self,
        env: gym.Env,
        randomize: bool = True,
        rand_range: float = 0.20,
        mismatch_level: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.randomize      = randomize
        self.rand_range     = rand_range
        self.mismatch_level = mismatch_level

    def _apply(self, params: Dict[str, float]) -> None:
        base = self.env.unwrapped
        for k, v in params.items():
            setattr(base, k, float(v))

    def reset(self, **kwargs):
        if self.randomize:
            new_params = {
                k: v * (1.0 + np.random.uniform(-self.rand_range, self.rand_range))
                for k, v in NOMINAL_PARAMS.items()
            }
        else:
            new_params = {
                k: v * (1.0 + self.mismatch_level)
                for k, v in NOMINAL_PARAMS.items()
            }
        self._apply(new_params)
        return self.env.reset(**kwargs)


# Policy/value network
def layer_init(layer: nn.Linear, std: float = np.sqrt(2),
               bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Shared Actor-Critic MLP.
    128×128 hidden units (larger than swing-only; needed for balance task).
    tanh activations + orthogonal init (CleanRL convention).
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)),     nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)),     nn.Tanh(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor,
                              action: Optional[torch.Tensor] = None):
        from torch.distributions.categorical import Categorical
        logits = self.actor(x)
        probs  = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    @torch.no_grad()
    def act_single(self, obs: np.ndarray, device: str = "cpu") -> int:
        from torch.distributions.categorical import Categorical
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return Categorical(logits=self.actor(x)).sample().item()


# Env factory
def make_balance_env(
    seed: int,
    idx: int,
    randomize: bool = True,
    rand_range: float = 0.20,
    mismatch_level: float = 0.0,
    max_episode_steps: int = 1000,
):
    """
    Full env stack:
      TimeLimit(DomainRandomized(AcrobotBalance(Acrobot-v1)))
    """
    def thunk():
        # Add wrappers in one place so train/eval build the same stack.
        base = gym.make("Acrobot-v1", max_episode_steps=max_episode_steps)
        env  = AcrobotBalanceEnv(base)
        env  = DomainRandomizedAcrobot(env, randomize=randomize,
                                        rand_range=rand_range,
                                        mismatch_level=mismatch_level)
        env  = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk
