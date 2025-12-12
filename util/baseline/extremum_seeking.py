from . import register_baseline
from ..trace_collec import trace_filter, flatten_leaves
import numpy as np

@register_baseline
class ExtremumSeeking:
    def __init__(self, state_cfg, initial_cmd, reward_cfg=None):
        self.state_cfg = state_cfg
        self.reward_cfg = reward_cfg if reward_cfg else {}

        self.u = np.array(initial_cmd.copy() if initial_cmd else [0.5, 0.5], dtype=float)
        self.initial_cmd = self.u.copy()

        self.step_size = np.array([0.1] * self.u.size, dtype=float)
        self.min_step_size = np.array([0.01] * self.u.size, dtype=float)
        self.max_step_size = np.array([0.5] * self.u.size, dtype=float)

        self.delta = np.array([0.08] * self.u.size, dtype=float)
        self.min_delta = np.array([0.01] * self.u.size, dtype=float)
        self.max_delta = np.array([0.25] * self.u.size, dtype=float)

        self.noise_std = np.array([0.02] * self.u.size, dtype=float)

        self.u_min = np.array([0.0] * self.u.size, dtype=float)
        self.u_max = np.array([1.0] * self.u.size, dtype=float)

        seed = None
        if isinstance(state_cfg, dict):
            seed = state_cfg.get("seed", None)
        self.rng = np.random.default_rng(seed)

        self.d = self._sample_dir()
        self.phase = 0
        self.J_plus = None
        self.J_mid_prev = None

        self.improvement_history = []
        self.window_size = 10

        self.iteration = 0

    def act(self, current_stats, **kwargs):
        self.iteration += 1

        reward = flatten_leaves(trace_filter(current_stats, self.reward_cfg))
        reward = np.asarray(reward, dtype=float)
        J_meas = float(np.mean(reward)) if reward.size > 0 else float("-inf")

        if self.phase == 0:
            self.J_plus = J_meas
            u_apply = self._clip_u(self.u - self.delta * self.d + self._noise())
            self.phase = 1
            return u_apply.tolist()

        J_minus = J_meas
        if self.J_plus is None or not np.isfinite(self.J_plus):
            self.J_plus = J_minus

        denom = 2.0 * np.maximum(self.delta, 1e-12)
        g = ((self.J_plus - J_minus) / denom) * self.d

        self.u = self._clip_u(self.u + self.step_size * g)

        J_mid = 0.5 * (self.J_plus + J_minus)
        if self.J_mid_prev is not None and np.isfinite(self.J_mid_prev) and np.isfinite(J_mid):
            improved = J_mid > self.J_mid_prev
            self._update_gains(improved)
        self.J_mid_prev = J_mid

        self.d = self._sample_dir()
        self.J_plus = None

        u_apply = self._clip_u(self.u + self.delta * self.d + self._noise())
        self.phase = 0
        return u_apply.tolist()

    def _sample_dir(self):
        d = self.rng.choice([-1.0, 1.0], size=self.u.size)
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else np.ones_like(d) / np.sqrt(d.size)

    def _noise(self):
        return self.rng.normal(loc=0.0, scale=self.noise_std, size=self.u.size)

    def _clip_u(self, u):
        u = np.asarray(u, dtype=float)
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    def _update_gains(self, improved):
        self.improvement_history.append(bool(improved))
        if len(self.improvement_history) > self.window_size:
            self.improvement_history.pop(0)

        if len(self.improvement_history) >= 5:
            rate = sum(self.improvement_history[-5:]) / 5.0
            if rate > 0.7:
                self.step_size = np.minimum(self.step_size * 1.05, self.max_step_size)
                self.delta = np.minimum(self.delta * 1.02, self.max_delta)
            elif rate < 0.3:
                self.step_size = np.maximum(self.step_size * 0.90, self.min_step_size)
                self.delta = np.maximum(self.delta * 0.95, self.min_delta)
