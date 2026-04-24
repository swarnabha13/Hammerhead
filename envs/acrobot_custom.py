"""
envs/acrobot_custom.py

AcrobotBalanceEnv — wraps Gymnasium's AcrobotEnv to:
  1. Use a dense reward (tip height + velocity penalty + upright bonus)
     instead of the original sparse {-1, 0} reward.
  2. Remove early termination — the agent must *hold* balance for the full
     episode, not just reach the upright state once.
  3. Support domain randomisation (DR): physics parameters are re-sampled
     from a uniform distribution at every reset().
  4. Support fixed-parameter evaluation at a chosen mismatch level.

Observation (unchanged from Gymnasium Acrobot):
  [cos θ₁, sin θ₁, cos θ₂, sin θ₂, dθ₁/dt, dθ₂/dt]

Tip height:
  h = -cos(θ₁) - cos(θ₁ + θ₂)   ∈ [-2, 2]
  h = +2 → fully upright (both links pointing up)
  h = -2 → fully hanging  (both links pointing down)
  The Gymnasium upright threshold is h > 1.0.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

# ── Nominal physics parameters (Gymnasium defaults) ───────────────────────────
NOMINAL_PARAMS: dict[str, float] = {
    "link_length_1":  1.0,
    "link_length_2":  1.0,
    "link_mass_1":    1.0,
    "link_mass_2":    1.0,
    "link_com_pos_1": 0.5,
    "link_com_pos_2": 0.5,
    "link_moi":       1.0,
}

# Keys we randomise / vary at evaluation
RANDOMISED_KEYS: list[str] = [
    "link_length_1",
    "link_length_2",
    "link_mass_1",
    "link_mass_2",
    "link_moi",
]

UPRIGHT_THRESHOLD: float = 1.0  # same as Gymnasium's termination threshold


def _make_base_env(params: dict[str, float], max_episode_steps: int) -> gym.Env:
    """Build a TimeLimit-wrapped AcrobotEnv with the given physics params."""
    env = AcrobotEnv(render_mode=None)
    for k, v in params.items():
        setattr(env, k, v)
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)


class AcrobotBalanceEnv(gym.Env):
    """
    Dense-reward, no-early-stop Acrobot with optional domain randomisation.

    Parameters
    ----------
    randomize : bool
        Resample physics at every reset() call.
    dr_range : float
        Half-width of the uniform DR distribution  (e.g. 0.20 → ±20 %).
    fixed_params : dict | None
        Fixed physics overrides for evaluation.  Ignored when randomize=True.
    max_episode_steps : int
        Episode length; truncated=True fires at this step count.
    vel_penalty_coef : float
        Weight of the angular-velocity penalty term.
    upright_bonus : float
        Per-step bonus when the tip height exceeds UPRIGHT_THRESHOLD.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        randomize: bool = False,
        dr_range: float = 0.20,
        fixed_params: dict[str, float] | None = None,
        max_episode_steps: int = 500,
        vel_penalty_coef: float = 0.05,
        upright_bonus: float = 0.5,
    ):
        super().__init__()
        self.randomize        = randomize
        self.dr_range         = dr_range
        self.fixed_params     = fixed_params or {}
        self.max_episode_steps = max_episode_steps
        self.vel_penalty_coef = vel_penalty_coef
        self.upright_bonus    = upright_bonus

        self._params = self._sample_params()
        self._env    = _make_base_env(self._params, max_episode_steps)

        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self._steps_upright = 0
        self._total_steps   = 0

    # ── parameter sampling ────────────────────────────────────────────────────
    def _sample_params(self) -> dict[str, float]:
        params = dict(NOMINAL_PARAMS)
        if self.randomize:
            for k in RANDOMISED_KEYS:
                scale = np.random.uniform(1.0 - self.dr_range, 1.0 + self.dr_range)
                params[k] = NOMINAL_PARAMS[k] * scale
        else:
            for k, v in self.fixed_params.items():
                params[k] = v
        return params

    # ── reward ────────────────────────────────────────────────────────────────
    @staticmethod
    def tip_height(obs: np.ndarray) -> float:
        """
        h = -cos θ₁ - cos(θ₁ + θ₂)
          = -cos_t1 - (cos_t1·cos_t2 - sin_t1·sin_t2)
        Range: [-2, 2]; maximum 2 when fully upright.
        """
        c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
        return float(-c1 - (c1 * c2 - s1 * s2))

    def _reward(self, obs: np.ndarray) -> tuple[float, bool]:
        h  = self.tip_height(obs)
        dt1, dt2 = float(obs[4]), float(obs[5])

        r_height   = (h + 2.0) / 4.0                          # normalised → [0, 1]
        r_vel      = -self.vel_penalty_coef * (dt1**2 + dt2**2)
        is_upright = h > UPRIGHT_THRESHOLD
        r_bonus    = self.upright_bonus if is_upright else 0.0

        return r_height + r_vel + r_bonus, is_upright

    # ── gym interface ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        self._params = self._sample_params()
        self._env    = _make_base_env(self._params, self.max_episode_steps)
        obs, info = self._env.reset(seed=seed, options=options)
        self._steps_upright = 0
        self._total_steps   = 0
        info["params"] = dict(self._params)
        return obs, info

    def step(self, action):
        obs, _orig_rew, terminated, truncated, info = self._env.step(action)

        reward, is_upright = self._reward(obs)
        terminated = False          # never terminate early

        self._steps_upright += int(is_upright)
        self._total_steps   += 1

        if truncated:
            info["upright_fraction"] = (
                self._steps_upright / self._total_steps if self._total_steps > 0 else 0.0
            )
            info["steps_upright"] = self._steps_upright

        info["tip_height"]  = self.tip_height(obs)
        info["is_upright"]  = is_upright
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def current_params(self) -> dict[str, float]:
        return dict(self._params)


# ── Factory for vectorised envs (CleanRL-style) ───────────────────────────────
def make_env(
    randomize: bool = True,
    dr_range: float = 0.20,
    fixed_params: dict[str, float] | None = None,
    max_episode_steps: int = 500,
    seed: int = 0,
    idx: int = 0,
    capture_video: bool = False,
    run_name: str = "run",
):
    def thunk():
        env = AcrobotBalanceEnv(
            randomize=randomize,
            dr_range=dr_range,
            fixed_params=fixed_params,
            max_episode_steps=max_episode_steps,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


# ── Evaluation helpers ────────────────────────────────────────────────────────
def build_eval_params(mismatch_fraction: float, direction: str = "positive") -> dict[str, float]:
    """
    Build fixed-params dict with all RANDOMISED_KEYS shifted by mismatch_fraction.

    direction : "positive" → all params × (1 + ε)
                "negative" → all params × (1 − ε)
                "random"   → each param independently ∈ U[1−ε, 1+ε]
    """
    params = dict(NOMINAL_PARAMS)
    rng = np.random.default_rng()
    for k in RANDOMISED_KEYS:
        if direction == "positive":
            scale = 1.0 + mismatch_fraction
        elif direction == "negative":
            scale = 1.0 - mismatch_fraction
        else:
            scale = float(rng.uniform(1.0 - mismatch_fraction, 1.0 + mismatch_fraction))
        params[k] = NOMINAL_PARAMS[k] * scale
    return params
