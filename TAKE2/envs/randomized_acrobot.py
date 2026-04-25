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
    ) -> None:
        super().__init__(render_mode=render_mode)
        assert 0.0 <= dr_range <= 1.0, "dr_range must be in [0, 1]"
        self.dr_range = dr_range
        self.fixed_mismatch = fixed_mismatch
        self._current_params: Dict[str, float] = {}
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
        # Resample params after parent reset so np_random is seeded
        self._resample_and_apply()
        info["params"] = self._current_params.copy()
        return obs, info

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
    render_mode    : only applied to env with idx == 0 (for visual inspection)
    """

    def thunk() -> gym.Env:
        _render = render_mode if idx == 0 else None
        env = RandomizedAcrobotEnv(
            render_mode=_render,
            dr_range=dr_range,
            fixed_mismatch=fixed_mismatch,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env

    return thunk
