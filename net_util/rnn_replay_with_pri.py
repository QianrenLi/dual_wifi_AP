from dataclasses import dataclass
from typing import List, Optional, Dict, Iterable, Tuple
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
    __slots__ = ("id", "obs", "actions", "rewards", "next_obs", "dones", "loss", "load_t")

    def __init__(self, eid: int, obs_np, act_np, rew_np, next_obs_np, done_np, device, init_loss: float = 1.0):
        self.id = eid
        self.obs = th.tensor(obs_np, device=device)             # [T, obs_dim]
        self.actions = th.tensor(act_np, device=device)         # [T, act_dim]
        self.rewards = th.tensor(rew_np, device=device)         # [T]
        self.next_obs = th.tensor(next_obs_np, device=device)   # [T, obs_dim]
        self.dones = th.tensor(done_np, device=device)          # [T]
        self.loss = float(init_loss)                            # scalar priority proxy
        self.load_t = 0                                         # cursor

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])

    def reset_cursor(self):
        self.load_t = 0

    def load(self):
        """Return one (o, a, r, no, d) at current cursor and advance; each with time axis [1, ...]."""
        if self.load_t >= self.length:
            raise StopIteration
        t = self.load_t
        self.load_t += 1
        # add time axis = 1 for RNN step: [1, 1, feat] after we unsqueeze batch later
        return (
            self.obs[t:t+1],       # [1, obs_dim]
            self.actions[t:t+1],   # [1, act_dim]
            self.rewards[t:t+1],   # [1]
            self.next_obs[t:t+1],  # [1, obs_dim]
            self.dones[t:t+1],     # [1]
        )

# -----------------------------
# RNNPriReplayBuffer Class
# -----------------------------
@register_buffer
@dataclass
class RNNPriReplayBuffer:
    """
    Replay buffer that stores entire episodes for RNN training.
    Adds:
      - max_episodes constraint with eviction,
      - per-episode loss/priority and weighted sampling.
    """
    episodes: List[Episode]
    device: str

    # prioritization/retention config
    max_episodes: int = 10000                  # cap; set large if you don't want tight limits
    evict_strategy: str = "low_priority"       # "low_priority" or "fifo"
    priority_alpha: float = 1.0                # exponent: weight = (eps + loss)^alpha
    priority_eps: float = 1e-3                 # stability: avoid zero weight
    _next_eid: int = 0                         # internal monotonic id
    beta = 0.9

    # ---------- Construction ----------
    @staticmethod
    def create(
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
        max_episodes: int = 10000,
        evict_strategy: str = "low_priority",
        priority_alpha: float = 1.0,
        priority_eps: float = 1e-3,
    ):
        return RNNPriReplayBuffer(
            episodes=[],
            device=device,
            max_episodes=max_episodes,
            evict_strategy=evict_strategy,
            priority_alpha=priority_alpha,
            priority_eps=priority_eps,
            _next_eid=0,
        )

    # ---------- Internals ----------
    def _evict_if_needed(self):
        if self.max_episodes <= 0:
            return
        while len(self.episodes) > self.max_episodes:
            if self.evict_strategy == "fifo":
                self.episodes.pop(0)
            elif self.evict_strategy == "low_priority":
                # drop the episode with the smallest loss; tie-breaker = oldest
                min_i = min(range(len(self.episodes)), key=lambda i: (self.episodes[i].loss, self.episodes[i].id))
                self.episodes.pop(min_i)
            else:
                raise ValueError(f"Unknown evict_strategy: {self.evict_strategy}")

    def _alloc_eid(self) -> int:
        eid = self._next_eid
        self._next_eid += 1
        return eid

    def _episode_weights(self, eps: Iterable[Episode]) -> np.ndarray:
        # weight ∝ (eps + loss)^alpha
        ls = np.asarray([max(self.priority_eps, float(e.loss)) for e in eps], dtype=np.float64)
        w = np.power(ls, float(self.priority_alpha))
        s = w.sum()
        return (w / s) if s > 0 else np.full_like(w, 1.0 / len(ls))

    def _weighted_sample_indices(self, k: int) -> List[int]:
        n = len(self.episodes)
        if n == 0:
            return []
        k = min(k, n)
        weights = self._episode_weights(self.episodes)
        # sample without replacement proportional to weights
        idx = np.arange(n)
        chosen = np.random.choice(idx, size=k, replace=False, p=weights)
        return chosen.tolist()

    # ---------- Add one episode ----------
    def add_episode(
        self,
        states,
        actions,
        rewards,
        network_output,
        enforce_last_done: bool = True,
        reward_agg: str = "sum",
        init_loss: Optional[float] = None,
    ):
        """
        Add one trace as a full episode to the buffer.
        init_loss: optional initial loss/priority for this episode (defaults to 1.0).
        """
        T = len(states)
        assert len(actions) == T and len(rewards) == T, "Episode arrays length mismatch."

        agg = _reward_agg_fn(reward_agg)
        rew_np = np.asarray([agg(_as_1d_float(r)) for r in rewards], dtype=np.float32)

        obs_np = np.stack([_as_1d_float(s) for s in states], axis=0).astype(np.float32)
        act_np = np.stack([_as_1d_float(a) for a in actions], axis=0).astype(np.float32)

        # next_obs: s_{t+1}, last = zeros
        if T > 1:
            next_obs_np = np.vstack([obs_np[1:], np.zeros((1, obs_np.shape[1]), dtype=np.float32)])
        else:
            next_obs_np = np.zeros((1, obs_np.shape[1]), dtype=np.float32)

        done_np = np.array([float(network_output[t].get("done", 0)) for t in range(T)], dtype=np.float32)
        if enforce_last_done:
            done_np[-1] = 1.0

        eid = self._alloc_eid()
        ep = Episode(eid, obs_np, act_np, rew_np, next_obs_np, done_np, self.device, init_loss=1.0 if init_loss is None else float(init_loss))
        self.episodes.append(ep)
        self._evict_if_needed()
        return eid

    # ---------- Update priorities ----------
    def update_episode_losses(self, ep_ids: Iterable[int], losses: Iterable[float] | float):
        """
        Update stored loss/priority for episodes by their IDs.
        Non-existent IDs are ignored silently.
        """
        if isinstance(losses, float):
            id2loss = {int(i): losses for i in ep_ids}
        else:
            id2loss = {int(i): float(l) for i, l in zip(ep_ids, losses)}
        for ep in self.episodes:
            if ep.id in id2loss:
                ep.loss = self.beta * ep.loss + id2loss[ep.id]

    # ---------- Sampling / Minibatching (RNN-aligned) ----------
    def get_minibatches(self, batch_size: int, shuffle: bool = True, prioritized: bool = True):
        """
        Yield minibatches for RNN-style training.
        - batch_size counts *episodes per batch*.
        - Each tensor has shape [B, 1, ...] (time axis length 1).
        - Returns: obs, act, rew, next_obs, done, info
          where info = {"ep_ids": List[int]} for mapping updates later.
        """
        num_eps = len(self.episodes)
        if num_eps == 0 or batch_size <= 0:
            return

        # 1) Choose which episodes enter the batch pool
        if prioritized:
            selected_idx = self._weighted_sample_indices(batch_size)
        else:
            idx = list(range(num_eps))
            if shuffle:
                random.shuffle(idx)
            selected_idx = idx[:batch_size]

        active_episodes: List[Episode] = [self.episodes[i] for i in selected_idx]
        for ep in active_episodes:
            ep.reset_cursor()

        ep_ids = [ep.id for ep in active_episodes]

        # 2) Stream batches one time-step at a time across active episodes
        while any(ep.load_t < ep.length for ep in active_episodes):
            O, A, R, NO, D = [], [], [], [], []

            for ep in active_episodes:
                try:
                    o, a, r, no, d = ep.load()  # each is [1, 1, dim]
                    O.append(o)   # list of [1, 1, dim]
                    A.append(a)
                    R.append(r)
                    NO.append(no)
                    D.append(d)
                except StopIteration:
                    pass

            if not O:
                break

            # concat over batch dimension => [B, 1, ...]
            obs_b      = th.cat(O,  dim=0)  # [B, 1, obs_dim]
            act_b      = th.cat(A,  dim=0)  # [B, 1, act_dim]
            rew_b      = th.cat(R,  dim=0)  # [B, 1, 1]
            next_obs_b = th.cat(NO, dim=0)  # [B, 1, obs_dim]
            done_b     = th.cat(D,  dim=0)  # [B, 1, 1]

            info = {"ep_ids": ep_ids}
            yield obs_b, act_b, rew_b, next_obs_b, done_b, info

    # ---------- Build from traces ----------
    @staticmethod
    def build_from_traces(
        traces,
        device: str = "cuda",
        reward_agg: str = "sum",
        n_envs: int = 1,
        max_episodes: int = 10000,
        evict_strategy: str = "low_priority",
        priority_alpha: float = 1.0,
        priority_eps: float = 1e-3,
        init_loss: Optional[float] = None,
        **kwargs,
    ):
        assert len(traces) >= 1, "No traces provided."
        first_states, first_actions, _, _ = traces[0]
        obs_dim = _as_1d_float(first_states[0]).size
        act_dim = _as_1d_float(first_actions[0]).size

        buf = RNNPriReplayBuffer.create(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            max_episodes=max_episodes,
            evict_strategy=evict_strategy,
            priority_alpha=priority_alpha,
            priority_eps=priority_eps,
        )

        for (states, actions, rewards, network_output) in traces:
            buf.add_episode(
                states, actions, rewards, network_output,
                enforce_last_done=True, reward_agg=reward_agg, init_loss=init_loss
            )
        return buf

    # ---------- In-place extension ----------
    def extend(
        self,
        traces,
        device: str = "cuda",
        reward_agg: str = "sum",
        init_loss: Optional[float] = None,
        **kwargs,
    ):
        for (states, actions, rewards, network_output) in traces:
            self.add_episode(
                states, actions, rewards, network_output,
                enforce_last_done=True, reward_agg=reward_agg, init_loss=init_loss
            )
