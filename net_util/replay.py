from dataclasses import dataclass
from net_util import register_buffer
import torch as th
import torch.nn.functional as F  # if you need it elsewhere
import numpy as np

# -----------------------------
# Helper
# -----------------------------
def _as_1d_float(x):
    """Coerce to 1-D float np.array without re-flattening dicts."""
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)

def _reward_agg_fn(reward_agg):
    if reward_agg == "sum":
        return lambda arr: float(np.asarray(arr, dtype=np.float32).sum())
    if reward_agg == "mean":
        return lambda arr: float(np.asarray(arr, dtype=np.float32).mean())
    if callable(reward_agg):
        return lambda arr: float(reward_agg(np.asarray(arr, dtype=np.float32)))
    raise ValueError("reward_agg must be 'sum', 'mean', or callable")

@register_buffer
@dataclass
class ReplayBuffer:
    # Storage: [capacity, n_envs, dim] except rewards/dones: [capacity, n_envs]
    obs: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    next_obs: th.Tensor
    dones: th.Tensor
    ptr: int = 0            # next write index (physical index in ring)
    size: int = 0           # number of valid items currently stored (<= capacity)

    # ---------- Construction ----------
    @staticmethod
    def create(buffer_size: int, obs_dim: int, act_dim: int, n_envs: int, device):
        obs      = th.zeros(buffer_size, n_envs, obs_dim, device=device)
        actions  = th.zeros(buffer_size, n_envs, act_dim, device=device)
        rewards  = th.zeros(buffer_size, n_envs, device=device)
        next_obs = th.zeros(buffer_size, n_envs, obs_dim, device=device)
        dones    = th.zeros(buffer_size, n_envs, device=device)
        # size must start at 0 for a cyclic queue
        return ReplayBuffer(obs, actions, rewards, next_obs, dones, 0, 0)

    # Convenience
    @property
    def capacity(self) -> int:
        return self.obs.shape[0]

    # ---------- Insertion (ring write) ----------
    def add(self, obs, action, reward, next_obs, done):
        """
        Insert a single transition (supports n_envs > 1 if inputs match).
        Overwrites oldest when full.
        """
        idx = self.ptr  # physical index to write
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = done

        # advance ring pointer
        self.ptr = (self.ptr + 1) % self.capacity
        # grow valid size until full
        if self.size < self.capacity:
            self.size += 1

    # ---------- Sampling / Minibatching ----------
    def _logical_to_physical(self, logical_idx: th.Tensor) -> th.Tensor:
        """
        Map logical indices in [0, size) to physical ring indices in [0, capacity).
        Oldest element has logical index 0.
        """
        if self.size == 0:
            return th.empty_like(logical_idx)
        start = (self.ptr - self.size) % self.capacity  # physical index of oldest
        return (start + logical_idx) % self.capacity

    def get_minibatches(self, batch_size: int):
        """
        Uniformly shuffle and yield minibatches from the VALID portion of the ring.
        Works regardless of ptr wrap position.
        """
        T = self.size
        if T == 0:
            return
        # logical shuffle [0..T-1]
        logical_perm = th.randperm(T, device=self.obs.device)
        # iterate in chunks
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            logical_idx = logical_perm[start:end]
            phys_idx = self._logical_to_physical(logical_idx)

            # gather rows by physical indices
            obs      = self.obs.index_select(0, phys_idx).view(end - start, -1)
            actions  = self.actions.index_select(0, phys_idx).view(end - start, -1)
            rewards  = self.rewards.index_select(0, phys_idx).view(end - start, -1)
            next_obs = self.next_obs.index_select(0, phys_idx).view(end - start, -1)
            dones    = self.dones.index_select(0, phys_idx).view(end - start, -1)
            yield obs, actions, rewards, next_obs, dones

    # ---------- Build from traces (with capacity cap as ring) ----------
    @staticmethod
    def build_from_traces(
        traces,                      # list of (states, actions, rewards, network_output)
        device="cuda",
        reward_agg="sum",            # "sum" | "mean" | callable(np.ndarray)->float
        buffer_max: int = 100_000,   # max capacity
        n_envs: int = 1,
        **kwargs,
    ):
        """
        Build a ring buffer of capacity `buffer_max` and insert steps from 'traces'.
        If the total steps exceed capacity, the oldest are overwritten (cyclic).
        """
        assert len(traces) >= 1, "No traces provided."
        agg = _reward_agg_fn(reward_agg)

        first_states, first_actions, _, _ = traces[0]
        obs_dim = _as_1d_float(first_states[0]).size
        act_dim = _as_1d_float(first_actions[0]).size

        cap = int(buffer_max)
        buf = ReplayBuffer.create(buffer_size=cap, obs_dim=obs_dim, act_dim=act_dim, n_envs=n_envs, device=device)

        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            assert len(actions) == T and len(rewards) == T and len(network_output) == T, "Trace lengths must match."

            for t in range(T):
                # build one-step tensors with [n_envs, dim]; here n_envs=1 by default
                obs_t = th.tensor(_as_1d_float(states[t]),  dtype=th.float32, device=device).unsqueeze(0)
                act_t = th.tensor(_as_1d_float(actions[t]), dtype=th.float32, device=device).unsqueeze(0)
                rew_t = th.tensor([agg(_as_1d_float(rewards[t]))], dtype=th.float32, device=device)

                # next obs = obs_{t+1} (or zeros at trajectory end)
                if t + 1 < T:
                    next_obs_t = th.tensor(_as_1d_float(states[t + 1]), dtype=th.float32, device=device).unsqueeze(0)
                else:
                    next_obs_t = th.zeros(1, obs_dim, device=device)

                done_t = th.tensor([float(network_output[t].get("done", 0))], dtype=th.float32, device=device)

                buf.add(obs=obs_t, action=act_t, reward=rew_t, next_obs=next_obs_t, done=done_t)

        return buf

    # ---------- In-place extension (no reallocation; ring append) ----------
    def extend(self, traces, device="cuda", reward_agg="sum", **kwargs):
        """
        Append new traces into the EXISTING ring buffer (overwrites oldest when full).
        """
        agg = _reward_agg_fn(reward_agg)

        # Validate dims against current buffer
        obs_dim = self.obs.shape[-1]
        act_dim = self.actions.shape[-1]

        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            assert len(actions) == T and len(rewards) == T and len(network_output) == T, "Trace lengths must match."
            # lightweight dim checks
            assert _as_1d_float(states[0]).size  == obs_dim, "obs_dim mismatch"
            assert _as_1d_float(actions[0]).size == act_dim, "act_dim mismatch"

            for t in range(T):
                obs_t = th.tensor(_as_1d_float(states[t]),  dtype=th.float32, device=device).unsqueeze(0)
                act_t = th.tensor(_as_1d_float(actions[t]), dtype=th.float32, device=device).unsqueeze(0)
                rew_t = th.tensor([agg(_as_1d_float(rewards[t]))], dtype=th.float32, device=device)

                if t + 1 < T:
                    next_obs_t = th.tensor(_as_1d_float(states[t + 1]), dtype=th.float32, device=device).unsqueeze(0)
                else:
                    next_obs_t = th.zeros(1, obs_dim, device=device)

                done_t = th.tensor([float(network_output[t].get("done", 0))], dtype=th.float32, device=device)

                self.add(obs=obs_t, action=act_t, reward=rew_t, next_obs=next_obs_t, done=done_t)
