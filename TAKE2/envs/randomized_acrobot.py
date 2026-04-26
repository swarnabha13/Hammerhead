"""
randomized_acrobot.py
---------------------
A subclass of Gymnasium's AcrobotEnv that supports:
  1. Domain Randomization (DR): at each episode reset, physical parameters are
     sampled uniformly from [nominal * (1 - dr_range), nominal * (1 + dr_range)].
  2. Fixed mismatch: a fixed scalar multiplier applied to ALL parameters, used
     during evaluation to simulate a specific sim-to-real gap level.

Physical parameters varied:
  - LINK_MASS_1    : mass of link 1 (kg)
  - LINK_MASS_2    : mass of link 2 (kg)
  - LINK_LENGTH_1  : length of link 1 (m)
  - LINK_LENGTH_2  : length of link 2 (m)
  - LINK_MOI       : moment of inertia for both links (kg·m²)

Physical parameters held fixed:
  - LINK_COM_POS_1 : center-of-mass position along link 1 (fraction of length)
  - LINK_COM_POS_2 : center-of-mass position along link 2 (fraction of length)

References:
  Sutton & Barto, Reinforcement Learning: An Introduction (2018)
  Gymnasium Acrobot: https://gymnasium.farama.org/environments/classic_control/acrobot/
"""

from __future__ import annotations

from math import pi
from typing import Dict, Optional, Callable

import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control.acrobot import AcrobotEnv


# ---------------------------------------------------------------------------
# Nominal parameter values (match Gymnasium's AcrobotEnv defaults exactly)
# ---------------------------------------------------------------------------
NOMINAL_PARAMS: Dict[str, float] = {
    "link_mass_1":    1.0,   # kg
    "link_mass_2":    1.0,   # kg
    "link_length_1":  1.0,   # m
    "link_length_2":  1.0,   # m
    "link_moi":       1.0,   # kg·m²
    "link_com_pos_1": 0.5,   # fraction (0→1)
    "link_com_pos_2": 0.5,   # fraction (0→1)
}

# Parameters we intentionally vary (com positions are kept at nominal because
# they are less physically interpretable as a "mismatch" metric)
VARIED_PARAMS = ["link_mass_1", "link_mass_2", "link_length_1", "link_length_2", "link_moi"]
MAX_EPISODE_STEPS = 500
BALANCE_HEIGHT_THRESHOLD = 1.75
BALANCE_VELOCITY_THRESHOLD = 1.25
BALANCE_HOLD_STEPS = 50


class RandomizedAcrobotEnv(AcrobotEnv):
    """
    Acrobot with configurable domain randomization for robust policy training.

    Usage
    -----
    # Training mode: domain randomize all varied params ±10% each episode
    env = RandomizedAcrobotEnv(dr_range=0.10)

    # Evaluation mode: all params shifted +10% (fixed mismatch = +0.10)
    env = RandomizedAcrobotEnv(fixed_mismatch=0.10)

    # Nominal baseline (no randomization, no mismatch)
    env = RandomizedAcrobotEnv()
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dr_range: float = 0.0,
        fixed_mismatch: float = 0.0,
        balance_reset_prob: float = 0.0,
    ) -> None:
        super().__init__(render_mode=render_mode)
        assert 0.0 <= dr_range <= 1.0, "dr_range must be in [0, 1]"
        assert 0.0 <= balance_reset_prob <= 1.0, "balance_reset_prob must be in [0, 1]"
        self.dr_range = dr_range
        self.fixed_mismatch = fixed_mismatch
        self.balance_reset_prob = balance_reset_prob
        self._current_params: Dict[str, float] = {}
        self._balance_steps = 0
        # Apply initial parameters
        self._resample_and_apply()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resample_and_apply(self) -> None:
        params: Dict[str, float] = {}
        for key, nominal in NOMINAL_PARAMS.items():
            if key in VARIED_PARAMS:
                if self.fixed_mismatch != 0.0:
                    # Evaluation: deterministic mismatch (e.g. +10% or -10%)
                    params[key] = nominal * (1.0 + self.fixed_mismatch)
                elif self.dr_range > 0.0:
                    # Training: uniform domain randomization
                    mult = self.np_random.uniform(
                        1.0 - self.dr_range, 1.0 + self.dr_range
                    )
                    params[key] = nominal * mult
                else:
                    params[key] = nominal
            else:
                # Non-varied parameters stay at nominal
                params[key] = nominal
        self._current_params = params
        self._apply_params(params)

    def _apply_params(self, params: Dict[str, float]) -> None:
        """Set instance-level physics attributes (override class defaults)."""
        self.LINK_MASS_1    = params["link_mass_1"]
        self.LINK_MASS_2    = params["link_mass_2"]
        self.LINK_LENGTH_1  = params["link_length_1"]
        self.LINK_LENGTH_2  = params["link_length_2"]
        self.LINK_MOI       = params["link_moi"]
        self.LINK_COM_POS_1 = params["link_com_pos_1"]
        self.LINK_COM_POS_2 = params["link_com_pos_2"]

    # ------------------------------------------------------------------
    # Gym API overrides
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Call parent reset first (sets self.np_random if seed provided)
        obs, info = super().reset(seed=seed, options=options)
        self._balance_steps = 0
        if self.balance_reset_prob > 0.0 and self.np_random.random() < self.balance_reset_prob:
            self.state = np.array([
                pi + self.np_random.uniform(-0.25, 0.25),
                self.np_random.uniform(-0.25, 0.25),
                self.np_random.uniform(-0.5, 0.5),
                self.np_random.uniform(-0.5, 0.5),
            ])
            obs = self._get_ob()
        # Resample params after parent reset so np_random is seeded
        self._resample_and_apply()
        info["params"] = self._current_params.copy()
        return obs, info

    def _tip_height(self) -> float:
        theta_1, theta_2 = self.state[0], self.state[1]
        return float(-np.cos(theta_1) - np.cos(theta_1 + theta_2))

    def _is_balanced(self) -> bool:
        height = self._tip_height()
        theta_dot_1, theta_dot_2 = self.state[2], self.state[3]
        return (
            height >= BALANCE_HEIGHT_THRESHOLD
            and abs(theta_dot_1) <= BALANCE_VELOCITY_THRESHOLD
            and abs(theta_dot_2) <= BALANCE_VELOCITY_THRESHOLD
        )

    def step(self, a):
        obs, _, _, truncated, info = super().step(a)

        if self._is_balanced():
            self._balance_steps += 1
        else:
            self._balance_steps = 0

        height = self._tip_height()
        theta_dot_1, theta_dot_2 = self.state[2], self.state[3]
        velocity_sq = theta_dot_1**2 + theta_dot_2**2
        upright_score = (height + 2.0) / 4.0
        high_tip_score = max(0.0, (height - 1.0) / 1.0)
        slow_upright_score = high_tip_score * np.exp(-0.35 * velocity_sq)
        torque = self.AVAIL_TORQUE[int(a)]
        torque_penalty = 0.01 * torque**2
        hold_progress = self._balance_steps / BALANCE_HOLD_STEPS

        reward = (
            2.0 * upright_score
            + 4.0 * high_tip_score
            + 6.0 * slow_upright_score
            + 8.0 * hold_progress
            - 0.05 * velocity_sq
            - torque_penalty
            - 0.05
        )

        terminated = self._balance_steps >= BALANCE_HOLD_STEPS
        if terminated:
            reward += 100.0
        info["tip_height"] = height
        info["balanced"] = self._balance_steps > 0
        info["balance_steps"] = self._balance_steps

        return obs, reward, terminated, truncated, info

    @property
    def current_params(self) -> Dict[str, float]:
        """Return a copy of the currently active physical parameters."""
        return self._current_params.copy()


# ---------------------------------------------------------------------------
# Vectorized-env factory (CleanRL style)
# ---------------------------------------------------------------------------

def make_env(
    seed: int,
    idx: int,
    dr_range: float = 0.0,
    fixed_mismatch: float = 0.0,
    balance_reset_prob: float = 0.0,
    render_mode: Optional[str] = None,
) -> Callable[[], gym.Env]:
    """
    Returns a thunk (zero-arg callable) that creates one Acrobot environment.
    Designed for use with gymnasium.vector.SyncVectorEnv.

    Parameters
    ----------
    seed           : base random seed (each env gets seed + idx)
    idx            : environment index (used for seeding and conditional render)
    dr_range       : domain randomization range for training
    fixed_mismatch : fixed mismatch level for evaluation (0 = nominal)
    balance_reset_prob : probability of starting near upright during training
    render_mode    : only applied to env with idx == 0 (for visual inspection)
    """

    def thunk() -> gym.Env:
        _render = render_mode if idx == 0 else None
        env = RandomizedAcrobotEnv(
            render_mode=_render,
            dr_range=dr_range,
            fixed_mismatch=fixed_mismatch,
            balance_reset_prob=balance_reset_prob,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env

    return thunk
