from . import register_baseline
from ..trace_collec import trace_filter, flatten_leaves
import numpy as np

@register_baseline
class ExtremumSeeking:
    def __init__(self, state_cfg, initial_cmd, reward_cfg=None):
        self.u = initial_cmd.copy() if initial_cmd else [0.5, 0.5]
        self.initial_cmd = self.u.copy()
        self.state_cfg = state_cfg

        self.reward_cfg = reward_cfg if reward_cfg else {}

        self.step_size = 0.1
        self.min_step_size = 0.01
        self.max_step_size = 0.5
        self.momentum = 0.9
        self.direction = np.ones(len(self.u), dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.J_prev = -float('inf')

        self.improvement_history = []
        self.window_size = 10

        self.u_min = [0.0, 0.0]
        self.u_max = [1.0, 1.0]

        self.iteration = 0

    def act(self, current_stats, **kwargs):
        self.iteration += 1

        reward = flatten_leaves(trace_filter(current_stats, self.reward_cfg))

        J = np.mean(reward)

        if self.J_prev != -float('inf'):
            improved = J > self.J_prev
            self._update_step_size(improved)

            if improved:
                self.direction = self.momentum * self.direction
            else:
                if np.random.rand() < 0.5:
                    delta = np.random.randn(len(self.u))
                    if np.linalg.norm(delta) > 1e-8:
                        delta /= np.linalg.norm(delta)
                    self.direction = -self.direction + (1 - self.momentum) * delta
                else:
                    self.direction = -self.direction

            if np.linalg.norm(self.direction) > 1e-8:
                self.direction = self.direction / np.linalg.norm(self.direction)
            u_array = np.array(self.u) + self.step_size * self.direction
            self.u = self._clip_u(u_array)

        self.J_prev = J
        return self.u.copy()

    def _extract_outage_rates(self, current_stats):
        outage_rates = []

        def _extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'outage_rate' and isinstance(value, (int, float)):
                        outage_rates.append(float(value))
                    _extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    _extract_recursive(item)

        _extract_recursive(current_stats)
        return outage_rates

    def _clip_u(self, u):
        u_array = np.array(u) if not isinstance(u, np.ndarray) else u
        clipped = np.clip(u_array, self.u_min, self.u_max)
        return clipped.tolist()

    def _update_step_size(self, improved):
        self.improvement_history.append(improved)
        if len(self.improvement_history) > self.window_size:
            self.improvement_history.pop(0)

        if len(self.improvement_history) >= 3:
            recent_rate = sum(self.improvement_history[-3:]) / 3.0
            if recent_rate > 0.7:
                self.step_size = min(self.step_size * 1.1, self.max_step_size)
            elif recent_rate < 0.3:
                self.step_size = max(self.step_size * 0.9, self.min_step_size)
