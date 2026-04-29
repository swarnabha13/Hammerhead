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


# Nominal parameters from Gymnasium's AcrobotEnv.
NOMINAL_PARAMS: Dict[str, float] = {
    "link_mass_1":    1.0,   # kg
    "link_mass_2":    1.0,   # kg
    "link_length_1":  1.0,   # m
    "link_length_2":  1.0,   # m
    "link_moi":       1.0,   # kg·m²
    "link_com_pos_1": 0.5,   # fraction (0→1)
    "link_com_pos_2": 0.5,   # fraction (0→1)
}

# I kept COM positions fixed; changing lengths/masses/inertia gave a clearer
# mismatch sweep without making the parameter space too broad.
VARIED_PARAMS = ["link_mass_1", "link_mass_2", "link_length_1", "link_length_2", "link_moi"]
MAX_EPISODE_STEPS = 1000
TARGET_HEIGHT = 1.0
PHASE_ANGLE_THRESHOLD = np.deg2rad(10.0)
BALANCE_LINK_ANGLE_THRESHOLD = PHASE_ANGLE_THRESHOLD
CAPTURE_ANGLE_THRESHOLD = np.deg2rad(55.0)
CAPTURE_HEIGHT_THRESHOLD = 0.25
BALANCE_VELOCITY_THRESHOLD = 1.5
HOLD_PROGRESS_STEPS = MAX_EPISODE_STEPS
HOLD_REWARD_RAMP_STEPS = 200
BALANCE_RESET_ANGLE_RANGE = np.deg2rad(4.0)
BALANCE_RESET_VELOCITY_RANGE = 0.05
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
        self._lqr_gain: Optional[np.ndarray] = None
        self._lqr_params_signature: Optional[tuple[float, ...]] = None
        # Apply a nominal/randomized set immediately so the first reset is valid.
        self._resample_and_apply()

    # Internal helpers

    def _resample_and_apply(self) -> None:
        params: Dict[str, float] = {}
        for key, nominal in NOMINAL_PARAMS.items():
            if key in VARIED_PARAMS:
                if self.fixed_mismatch != 0.0:
                    # Evaluation uses one deterministic shift for all varied params.
                    params[key] = nominal * (1.0 + self.fixed_mismatch)
                elif self.dr_range > 0.0:
                    # Training samples each varied parameter independently.
                    mult = self.np_random.uniform(
                        1.0 - self.dr_range, 1.0 + self.dr_range
                    )
                    params[key] = nominal * mult
                else:
                    params[key] = nominal
            else:
                # Everything outside VARIED_PARAMS stays at nominal.
                params[key] = nominal
        self._current_params = params
        self._apply_params(params)

    def _apply_params(self, params: Dict[str, float]) -> None:
        """Set instance-level physics attributes."""
        self.LINK_MASS_1    = params["link_mass_1"]
        self.LINK_MASS_2    = params["link_mass_2"]
        self.LINK_LENGTH_1  = params["link_length_1"]
        self.LINK_LENGTH_2  = params["link_length_2"]
        self.LINK_MOI       = params["link_moi"]
        self.LINK_COM_POS_1 = params["link_com_pos_1"]
        self.LINK_COM_POS_2 = params["link_com_pos_2"]
        self._lqr_gain = None
        self._lqr_params_signature = None

    # Gym API overrides

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Parent reset is what gives us Gymnasium's seeded RNG.
        obs, info = super().reset(seed=seed, options=options)
        self._balance_steps = 0
        self._episode_max_balance_steps = 0
        self._episode_balanced_steps = 0
        self._episode_upright_steps = 0
        self._episode_steps = 0
        if self.balance_reset_prob > 0.0 and self.np_random.random() < self.balance_reset_prob:
            link_1_abs = pi + self.np_random.uniform(
                -BALANCE_RESET_ANGLE_RANGE,
                BALANCE_RESET_ANGLE_RANGE,
            )
            link_2_abs = pi + self.np_random.uniform(
                -BALANCE_RESET_ANGLE_RANGE,
                BALANCE_RESET_ANGLE_RANGE,
            )
            theta_2 = self._angle_normalize(link_2_abs - link_1_abs)
            self.state = np.array([
                link_1_abs,
                theta_2,
                self.np_random.uniform(-BALANCE_RESET_VELOCITY_RANGE, BALANCE_RESET_VELOCITY_RANGE),
                self.np_random.uniform(-BALANCE_RESET_VELOCITY_RANGE, BALANCE_RESET_VELOCITY_RANGE),
            ])
            obs = self._get_ob()
        # Resample after parent reset so np_random is ready.
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
                self._tip_height() / self._max_reach(),
                min(1.0, self._balance_steps / HOLD_PROGRESS_STEPS),
            ],
            dtype=np.float32,
        )

    def _max_reach(self) -> float:
        return float(self.LINK_LENGTH_1 + self.LINK_LENGTH_2)

    def _tip_height(self) -> float:
        theta_1, theta_2 = self.state[0], self.state[1]
        return float(
            -self.LINK_LENGTH_1 * np.cos(theta_1)
            - self.LINK_LENGTH_2 * np.cos(theta_1 + theta_2)
        )

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
        max_reach = self._max_reach()
        height_score = np.clip((height + max_reach) / (2.0 * max_reach), 0.0, 1.0)
        upright_height_score = np.clip(height / max_reach, 0.0, 1.0)
        both_links_score = np.exp(-8.0 * angle_error_sq)
        slow_upright_score = both_links_score * np.exp(-0.75 * velocity_sq)
        hold_progress = min(1.0, self._balance_steps / HOLD_REWARD_RAMP_STEPS)
        return float(
            20.0 * height_score
            + 40.0 * upright_height_score
            + 220.0 * both_links_score
            + 360.0 * slow_upright_score
            + 320.0 * hold_progress
            - 0.08 * velocity_sq
        )

    def _link_2_in_balance_phase(self) -> bool:
        _, link_2_error = self._upright_errors()
        return abs(link_2_error) <= PHASE_ANGLE_THRESHOLD

    def _in_capture_phase(self) -> bool:
        height = self._tip_height()
        link_1_error, link_2_error = self._upright_errors()
        near_upright = (
            abs(link_1_error) <= CAPTURE_ANGLE_THRESHOLD
            and abs(link_2_error) <= CAPTURE_ANGLE_THRESHOLD
        )
        return height >= CAPTURE_HEIGHT_THRESHOLD or near_upright or self._balance_steps > 0

    def should_use_balance_controller(self) -> bool:
        return self._in_capture_phase()

    def _state_to_lqr_error(self, state: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self._angle_normalize(float(state[0]) - pi),
                self._angle_normalize(float(state[1])),
                float(state[2]),
                float(state[3]),
            ],
            dtype=np.float64,
        )

    def _simulate_one_step(self, state: np.ndarray, torque: float) -> np.ndarray:
        saved_state = np.array(self.state, dtype=np.float64).copy()
        saved_balance_steps = self._balance_steps
        saved_last_height = self._last_height
        saved_torque = self.AVAIL_TORQUE

        self.state = np.array(state, dtype=np.float64).copy()
        self.AVAIL_TORQUE = np.array([torque], dtype=np.float32)
        AcrobotEnv.step(self, 0)
        next_state = np.array(self.state, dtype=np.float64).copy()

        self.state = saved_state
        self._balance_steps = saved_balance_steps
        self._last_height = saved_last_height
        self.AVAIL_TORQUE = saved_torque
        return next_state

    def _get_lqr_gain(self) -> np.ndarray:
        signature = tuple(round(self._current_params[key], 8) for key in VARIED_PARAMS)
        if self._lqr_gain is not None and self._lqr_params_signature == signature:
            return self._lqr_gain

        equilibrium = np.array([pi, 0.0, 0.0, 0.0], dtype=np.float64)
        base_next = self._state_to_lqr_error(self._simulate_one_step(equilibrium, 0.0))
        eps = 1e-5

        a_mat = np.zeros((4, 4), dtype=np.float64)
        for idx in range(4):
            plus_state = equilibrium.copy()
            minus_state = equilibrium.copy()
            plus_state[idx] += eps
            minus_state[idx] -= eps
            plus_next = self._state_to_lqr_error(self._simulate_one_step(plus_state, 0.0))
            minus_next = self._state_to_lqr_error(self._simulate_one_step(minus_state, 0.0))
            a_mat[:, idx] = (plus_next - minus_next) / (2.0 * eps)

        b_plus = self._state_to_lqr_error(self._simulate_one_step(equilibrium, eps))
        b_minus = self._state_to_lqr_error(self._simulate_one_step(equilibrium, -eps))
        b_mat = ((b_plus - b_minus) / (2.0 * eps)).reshape(4, 1)
        if np.linalg.norm(b_mat) < 1e-10:
            b_mat = (b_plus - base_next).reshape(4, 1) / eps

        q_mat = np.diag([180.0, 120.0, 18.0, 10.0])
        r_mat = np.array([[0.08]], dtype=np.float64)
        p_mat = q_mat.copy()
        for _ in range(500):
            bt_p = b_mat.T @ p_mat
            gain = np.linalg.solve(r_mat + bt_p @ b_mat, bt_p @ a_mat)
            next_p = q_mat + a_mat.T @ p_mat @ (a_mat - b_mat @ gain)
            if np.max(np.abs(next_p - p_mat)) < 1e-9:
                p_mat = next_p
                break
            p_mat = next_p

        self._lqr_gain = np.linalg.solve(r_mat + b_mat.T @ p_mat @ b_mat, b_mat.T @ p_mat @ a_mat)
        self._lqr_params_signature = signature
        return self._lqr_gain

    def _lqr_action(self) -> int:
        error = self._state_to_lqr_error(np.array(self.state, dtype=np.float64))
        gain = self._get_lqr_gain()
        torque = float(np.clip(-(gain @ error)[0], TORQUE_VALUES[0], TORQUE_VALUES[-1]))
        return int(np.argmin(np.abs(TORQUE_VALUES - torque)))

    def _one_step_mpc_action(self) -> int:
        """
        One-step stabilizer over the discrete torque set.

        This is the teacher used around the capture/upright region.
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

    def expert_action(self) -> int:
        """Teacher action used during BC/DAgger data collection."""
        if self._in_capture_phase():
            return self._lqr_action()
        return self._one_step_mpc_action()

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
            self._episode_max_balance_steps = max(
                self._episode_max_balance_steps,
                self._balance_steps,
            )
            self._episode_balanced_steps += 1
        else:
            self._balance_steps = 0

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
        both_links_score = np.exp(-8.0 * angle_error_sq)
        max_reach = self._max_reach()
        height_score = np.clip((height + max_reach) / (2.0 * max_reach), 0.0, 1.0)
        upright_height_score = np.clip(height / max_reach, 0.0, 1.0)
        target_bonus = 1.0 if height >= TARGET_HEIGHT else 0.0
        above_target_score = max(0.0, height - TARGET_HEIGHT)
        swing_progress = np.clip(height_delta, -0.25, 0.25)
        link_2_abs_velocity = link_2_velocity
        link_2_motion = np.clip(abs(link_2_abs_velocity) / 4.0, 0.0, 1.0)
        link_2_height_score = (np.cos(link_2_error) + 1.0) / 2.0
        slow_upright_score = both_links_score * np.exp(-0.75 * velocity_sq)
        torque = self.AVAIL_TORQUE[int(a)]
        torque_penalty = 0.001 * torque**2
        hold_progress = min(1.0, self._balance_steps / HOLD_REWARD_RAMP_STEPS)
        hold_progress_sq = hold_progress * hold_progress
        in_balance_phase = self._link_2_in_balance_phase()
        in_capture_phase = self._in_capture_phase()
        if in_balance_phase:
            self._episode_upright_steps += 1

        if in_capture_phase:
            reward = (
                25.0 * height_score
                + 60.0 * upright_height_score
                + 40.0 * target_bonus
                + 6.0 * above_target_score
                + 50.0 * link_1_score
                + 50.0 * link_2_score
                + 260.0 * both_links_score
                + 420.0 * slow_upright_score
                + 280.0 * float(is_balanced)
                + 500.0 * hold_progress
                + 800.0 * hold_progress_sq
                - 0.75 * velocity_sq
                - 9.0 * (abs(link_1_error) + abs(link_2_error))
                - 1.2 * target_bonus * velocity_sq
                - torque_penalty
                + 1.0
            )
        else:
            reward = (
                8.0 * link_2_height_score
                + 14.0 * height_score
                + 6.0 * swing_progress
                + 2.0 * link_2_motion * (1.0 - link_2_height_score)
                + 2.0 * link_1_score
                - 0.01 * velocity_sq
                - torque_penalty
                - 0.05
            )

        terminated = False
        obs = self._get_ob()
        info["tip_height"] = height
        info["target_reached"] = height >= TARGET_HEIGHT
        info["link_1_upright_error"] = link_1_error
        info["link_2_upright_error"] = link_2_error
        info["phase"] = "capture" if in_capture_phase else "swing_up"
        info["upright"] = in_balance_phase
        info["capture"] = in_capture_phase
        info["balanced"] = is_balanced
        info["balance_steps"] = self._balance_steps
        info["episode_max_balance_steps"] = self._episode_max_balance_steps
        info["episode_balanced_steps"] = self._episode_balanced_steps
        info["episode_upright_steps"] = self._episode_upright_steps
        info["hold_score_fraction"] = self._episode_max_balance_steps / MAX_EPISODE_STEPS
        info["balance_time_fraction"] = self._episode_balanced_steps / max(1, self._episode_steps)
        info["upright_time_fraction"] = self._episode_upright_steps / max(1, self._episode_steps)

        return obs, reward, terminated, truncated, info

    @property
    def current_params(self) -> Dict[str, float]:
        """Return a copy of the active physical parameters."""
        return self._current_params.copy()


# Vectorized-env factory

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
