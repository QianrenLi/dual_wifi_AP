from dataclasses import dataclass
from net_util import register_buffer
import torch as th
import numpy as np
import random

# -----------------------------
# Helper Functions
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

# -----------------------------
# Episode Class
# -----------------------------
class Episode:
    def __init__(self, obs_np, act_np, rew_np, next_obs_np, done_np, device):
        self.obs = th.tensor(obs_np, device=device)
        self.actions = th.tensor(act_np, device=device)
        self.rewards = th.tensor(rew_np, device=device)
        self.next_obs = th.tensor(next_obs_np, device=device)
        self.dones = th.tensor(done_np, device=device)
        self.load_t = 0  # cursor

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])

    def reset_cursor(self):
        self.load_t = 0

    def load(self):
        """Return one (o, a, r, no, d) at current cursor and advance."""
        if self.load_t >= self.length:
            raise StopIteration
        t = self.load_t
        self.load_t += 1
        return (
            self.obs[t:t+1],       # [1, obs_dim]
            self.actions[t:t+1],   # [1, act_dim]
            self.rewards[t:t+1],   # [1]
            self.next_obs[t:t+1],  # [1, obs_dim]
            self.dones[t:t+1],     # [1]
        )

# -----------------------------
# RNNReplayBuffer Class
# -----------------------------
@register_buffer
@dataclass
class RNNReplayBuffer:
    """
    A replay buffer for storing episodes of (obs, action, reward, next_obs, done) for RNN-style training.
    No capacity limits, stores episodes indefinitely.
    """
    episodes: list  # List of Episode objects
    device: str

    # ---------- Construction ----------
    @staticmethod
    def create(obs_dim: int, act_dim: int, device="cuda"):
        return RNNReplayBuffer(
            episodes=[],
            device=device,
        )

    # ---------- Add one episode ----------
    def add_episode(self, states, actions, rewards, network_output, enforce_last_done=True, reward_agg="sum"):
        """
        Add one trace as a full episode to the buffer.
        """
        T = len(states)
        assert len(actions) == T and len(rewards) == T, "Episode arrays length mismatch."

        # Aggregate rewards (if necessary)
        agg = _reward_agg_fn(reward_agg)
        rew_np = np.asarray([agg(_as_1d_float(r)) for r in rewards], dtype=np.float32)

        # Convert states, actions, rewards into tensors
        obs_np = np.stack([_as_1d_float(s) for s in states], axis=0).astype(np.float32)
        act_np = np.stack([_as_1d_float(a) for a in actions], axis=0).astype(np.float32)

        # next_obs: s_{t+1}, last = zeros
        next_obs_np = np.vstack([obs_np[1:], np.zeros((1, obs_np.shape[1]), dtype=np.float32)]) if T > 1 else np.zeros((1, obs_np.shape[1]), dtype=np.float32)

        # Convert done flags from network_output to tensor for each time step
        done_np = np.array([float(network_output[t].get("done", 0)) for t in range(T)], dtype=np.float32)

        # Ensure the last timestep is done (if necessary)
        if enforce_last_done:
            done_np[-1] = 1.0  # last timestep is done

        # Create the episode and add it to the buffer
        ep = Episode(obs_np, act_np, rew_np, next_obs_np, done_np, self.device)
        self.episodes.append(ep)

    # ---------- Sampling / Minibatching (RNN-aligned) ----------
    def get_minibatches(self, batch_size: int, shuffle: bool = True):
        """
        Yield minibatches for RNN-style training. Each minibatch corresponds to
        one step from each active episode.
        - `batch_size` counts *episodes per batch*.
        - Each minibatch contains [B, 1, ...] where B is the batch size.
        """
        num_eps = len(self.episodes)
        if num_eps == 0 or batch_size <= 0:
            return

        # 1) Shuffle episodes
        idx = list(range(num_eps))
        if shuffle:
            random.shuffle(idx)

        # 2) Fill batch with episodes
        active_episodes = [self.episodes[i] for i in idx[:batch_size]]
        for ep in active_episodes:
            ep.reset_cursor()  # Reset cursor for each episode

        # 3) Stream minibatches until all episodes are finished
        while any(ep.load_t < ep.length for ep in active_episodes):
            obs_b, act_b, rew_b, next_obs_b, done_b = [], [], [], [], []

            # Collect one step from each active episode
            for ep in active_episodes:
                try:
                    o, a, r, no, d = ep.load()  # Each is [1, dim]
                    obs_b.append(o)
                    act_b.append(a)
                    rew_b.append(r)
                    next_obs_b.append(no)
                    done_b.append(d)
                except StopIteration:
                    pass  # Episode is finished

            # Yield minibatch
            yield th.cat(obs_b, dim=0), th.cat(act_b, dim=0), th.cat(rew_b, dim=0), th.cat(next_obs_b, dim=0), th.cat(done_b, dim=0)

    # ---------- Build from traces ----------
    @staticmethod
    def build_from_traces(
        traces, device="cuda", reward_agg="sum", n_envs: int = 1, **kwargs
    ):
        """
        Build a ring buffer from traces. It overwrites the oldest entries when full.
        """
        assert len(traces) >= 1, "No traces provided."

        first_states, first_actions, _, _ = traces[0]
        obs_dim = _as_1d_float(first_states[0]).size
        act_dim = _as_1d_float(first_actions[0]).size

        buf = RNNReplayBuffer.create(obs_dim=obs_dim, act_dim=act_dim, device=device)

        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            buf.add_episode(states, actions, rewards, network_output)
        return buf

    # ---------- In-place extension ----------
    def extend(self, traces, device="cuda", reward_agg="sum", **kwargs):
        """
        Append new traces into the existing buffer (overwrites oldest when full).
        """

        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            self.add_episode(states, actions, rewards, network_output)
