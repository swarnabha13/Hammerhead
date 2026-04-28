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
MAX_EPISODE_STEPS = 1000
MAX_TIP_HEIGHT = 2.0
TARGET_HEIGHT = 1.9
PHASE_ANGLE_THRESHOLD = np.deg2rad(10.0)
BALANCE_LINK_ANGLE_THRESHOLD = PHASE_ANGLE_THRESHOLD
BALANCE_VELOCITY_THRESHOLD = 1.5
BALANCE_HOLD_STEPS = 500
SUCCESS_BONUS = 12000.0
TORQUE_VALUES = np.array(
    [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    dtype=np.float32,
)
OBSERVATION_DIM = 10


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
        self.AVAIL_TORQUE = TORQUE_VALUES
        self.action_space = gym.spaces.Discrete(len(TORQUE_VALUES))
        high = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                self.MAX_VEL_1,
                self.MAX_VEL_2,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        low = -high
        low[-1] = 0.0
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        assert 0.0 <= dr_range <= 1.0, "dr_range must be in [0, 1]"
        assert 0.0 <= balance_reset_prob <= 1.0, "balance_reset_prob must be in [0, 1]"
        self.dr_range = dr_range
        self.fixed_mismatch = fixed_mismatch
        self.balance_reset_prob = balance_reset_prob
        self._current_params: Dict[str, float] = {}
        self._balance_steps = 0
        self._episode_max_balance_steps = 0
        self._episode_balanced_steps = 0
        self._episode_upright_steps = 0
        self._episode_steps = 0
        self._last_height = 0.0
        self._success_bonus_paid = False
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
        self._episode_max_balance_steps = 0
        self._episode_balanced_steps = 0
        self._episode_upright_steps = 0
        self._episode_steps = 0
        self._success_bonus_paid = False
        if self.balance_reset_prob > 0.0 and self.np_random.random() < self.balance_reset_prob:
            link_1_abs = pi + self.np_random.uniform(-0.15, 0.15)
            link_2_abs = pi + self.np_random.uniform(-0.15, 0.15)
            theta_2 = self._angle_normalize(link_2_abs - link_1_abs)
            self.state = np.array([
                link_1_abs,
                theta_2,
                self.np_random.uniform(-0.2, 0.2),
                self.np_random.uniform(-0.2, 0.2),
            ])
            obs = self._get_ob()
        # Resample params after parent reset so np_random is seeded
        self._resample_and_apply()
        obs = self._get_ob()
        self._last_height = self._tip_height()
        info["params"] = self._current_params.copy()
        return obs, info

    def _get_ob(self):
        theta_1, theta_2, theta_dot_1, theta_dot_2 = self.state
        theta_2_abs = theta_1 + theta_2
        return np.array(
            [
                np.cos(theta_1),
                np.sin(theta_1),
                np.cos(theta_2),
                np.sin(theta_2),
                theta_dot_1,
                theta_dot_2,
                np.cos(theta_2_abs),
                np.sin(theta_2_abs),
                self._tip_height() / 2.0,
                min(1.0, self._balance_steps / MAX_EPISODE_STEPS),
            ],
            dtype=np.float32,
        )

    def _tip_height(self) -> float:
        theta_1, theta_2 = self.state[0], self.state[1]
        return float(-np.cos(theta_1) - np.cos(theta_1 + theta_2))

    @staticmethod
    def _angle_normalize(angle: float) -> float:
        return float(((angle + pi) % (2 * pi)) - pi)

    def _upright_errors(self) -> tuple[float, float]:
        theta_1, theta_2 = self.state[0], self.state[1]
        link_1_error = self._angle_normalize(theta_1 - pi)
        link_2_error = self._angle_normalize(theta_1 + theta_2 - pi)
        return link_1_error, link_2_error

    def _balance_score(self) -> float:
        height = self._tip_height()
        theta_dot_1, theta_dot_2 = self.state[2], self.state[3]
        link_1_error, link_2_error = self._upright_errors()
        link_1_velocity = theta_dot_1
        link_2_velocity = theta_dot_1 + theta_dot_2
        angle_error_sq = link_1_error**2 + link_2_error**2
        velocity_sq = link_1_velocity**2 + link_2_velocity**2 + theta_dot_2**2
        target_progress = np.clip((height + 2.0) / (MAX_TIP_HEIGHT + 2.0), 0.0, 1.0)
        both_links_score = np.exp(-5.0 * angle_error_sq)
        slow_upright_score = both_links_score * np.exp(-0.7 * velocity_sq)
        target_bonus = 1.0 if height >= TARGET_HEIGHT else 0.0
        hold_progress = min(1.0, self._balance_steps / MAX_EPISODE_STEPS)
        return float(
            8.0 * target_progress
            + 20.0 * target_bonus
            + 260.0 * both_links_score
            + 420.0 * slow_upright_score
            + 180.0 * hold_progress
            - 0.12 * velocity_sq
        )

    def _link_2_in_balance_phase(self) -> bool:
        _, link_2_error = self._upright_errors()
        return abs(link_2_error) <= PHASE_ANGLE_THRESHOLD

    def should_use_balance_controller(self) -> bool:
        height = self._tip_height()
        link_1_error, link_2_error = self._upright_errors()
        near_upright = abs(link_1_error) < 1.35 and abs(link_2_error) < 1.35
        return height >= 0.25 or near_upright or self._balance_steps > 0

    def expert_action(self) -> int:
        """
        One-step model-predictive stabilizer over the discrete torque set.

        This is used as a near-upright safety/stabilization controller for the
        harder "swing up and hold both links upright" objective.
        """
        saved_state = np.array(self.state, dtype=np.float64).copy()
        saved_balance_steps = self._balance_steps
        saved_last_height = self._last_height

        best_action = 0
        best_score = -float("inf")
        for action in range(self.action_space.n):
            self.state = saved_state.copy()
            AcrobotEnv.step(self, action)
            score = self._balance_score() - 0.002 * float(self.AVAIL_TORQUE[action] ** 2)
            if score > best_score:
                best_score = score
                best_action = action

        self.state = saved_state
        self._balance_steps = saved_balance_steps
        self._last_height = saved_last_height
        return int(best_action)

    def _is_balanced(self) -> bool:
        height = self._tip_height()
        theta_dot_1, theta_dot_2 = self.state[2], self.state[3]
        link_1_error, link_2_error = self._upright_errors()
        link_1_velocity = theta_dot_1
        link_2_velocity = theta_dot_1 + theta_dot_2
        return (
            height >= TARGET_HEIGHT
            and abs(link_1_error) <= BALANCE_LINK_ANGLE_THRESHOLD
            and abs(link_2_error) <= BALANCE_LINK_ANGLE_THRESHOLD
            and abs(link_1_velocity) <= BALANCE_VELOCITY_THRESHOLD
            and abs(link_2_velocity) <= BALANCE_VELOCITY_THRESHOLD
            and abs(theta_dot_2) <= BALANCE_VELOCITY_THRESHOLD
        )

    def step(self, a):
        obs, _, _, truncated, info = super().step(a)
        self._episode_steps += 1

        is_balanced = self._is_balanced()
        if is_balanced:
            self._balance_steps += 1
            self._episode_balanced_steps += 1
        else:
            self._balance_steps = 0
        self._episode_max_balance_steps = max(
            self._episode_max_balance_steps, self._balance_steps
        )

        height = self._tip_height()
        height_delta = height - self._last_height
        self._last_height = height
        theta_dot_1, theta_dot_2 = self.state[2], self.state[3]
        link_1_error, link_2_error = self._upright_errors()
        link_1_velocity = theta_dot_1
        link_2_velocity = theta_dot_1 + theta_dot_2
        velocity_sq = link_1_velocity**2 + link_2_velocity**2 + theta_dot_2**2
        angle_error_sq = link_1_error**2 + link_2_error**2
        link_1_score = (np.cos(link_1_error) + 1.0) / 2.0
        link_2_score = (np.cos(link_2_error) + 1.0) / 2.0
        both_links_score = np.exp(-4.5 * angle_error_sq)
        full_upright_score = np.exp(-12.0 * angle_error_sq)
        target_progress = np.clip((height + 2.0) / (MAX_TIP_HEIGHT + 2.0), 0.0, 1.0)
        target_bonus = 1.0 if height >= TARGET_HEIGHT else 0.0
        above_target_score = max(0.0, height - TARGET_HEIGHT) / (MAX_TIP_HEIGHT - TARGET_HEIGHT)
        swing_progress = np.clip(height_delta, -0.25, 0.25)
        link_2_abs_velocity = link_2_velocity
        link_2_motion = np.clip(abs(link_2_abs_velocity) / 4.0, 0.0, 1.0)
        link_2_height_score = (np.cos(link_2_error) + 1.0) / 2.0
        slow_upright_score = both_links_score * np.exp(-0.65 * velocity_sq)
        torque = self.AVAIL_TORQUE[int(a)]
        torque_penalty = 0.001 * torque**2
        hold_progress = min(1.0, self._balance_steps / MAX_EPISODE_STEPS)
        in_balance_phase = self._link_2_in_balance_phase()
        if in_balance_phase:
            self._episode_upright_steps += 1

        # Smoothly blend swing-up shaping with stabilization shaping. A hard
        # phase switch around one link made it easy to visit the target without
        # discovering the much narrower both-links-upright, low-velocity basin.
        readiness = float(np.clip((height - 0.15) / 1.35, 0.0, 1.0))
        readiness = max(readiness, float(np.exp(-0.75 * angle_error_sq)))
        swing_reward = (
            9.0 * link_2_height_score
            + 12.0 * target_progress
            + 8.0 * swing_progress
            + 2.5 * link_2_motion * (1.0 - link_2_height_score)
            + 3.0 * link_1_score
            - 0.012 * velocity_sq
            - torque_penalty
            - 0.05
        )
        balance_reward = (
            18.0 * target_progress
            + 14.0 * target_bonus
            + 10.0 * above_target_score
            + 45.0 * link_1_score
            + 30.0 * link_2_score
            + 260.0 * both_links_score
            + 180.0 * full_upright_score
            + 420.0 * slow_upright_score
            + 65.0 * float(is_balanced)
            + 420.0 * hold_progress
            - 0.16 * velocity_sq
            - 3.5 * abs(link_1_error)
            - 2.5 * abs(link_2_error)
            - torque_penalty
        )
        reward = (1.0 - readiness) * swing_reward + readiness * balance_reward

        success = self._episode_max_balance_steps >= BALANCE_HOLD_STEPS
        success_just_reached = (
            self._balance_steps >= BALANCE_HOLD_STEPS
            and not self._success_bonus_paid
        )
        terminated = False
        if success_just_reached:
            reward += SUCCESS_BONUS
            self._success_bonus_paid = True
        obs = self._get_ob()
        info["tip_height"] = height
        info["target_reached"] = height >= TARGET_HEIGHT
        info["link_1_upright_error"] = link_1_error
        info["link_2_upright_error"] = link_2_error
        info["phase"] = "balance" if in_balance_phase else "swing_up"
        info["upright"] = in_balance_phase
        info["balanced"] = is_balanced
        info["success"] = success
        info["success_just_reached"] = success_just_reached
        info["balance_steps"] = self._balance_steps
        info["max_balance_steps"] = self._episode_max_balance_steps
        info["episode_balanced_steps"] = self._episode_balanced_steps
        info["episode_upright_steps"] = self._episode_upright_steps
        info["episode_steps"] = self._episode_steps
        info["balance_time_fraction"] = self._episode_balanced_steps / max(1, self._episode_steps)
        info["upright_time_fraction"] = self._episode_upright_steps / max(1, self._episode_steps)

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
