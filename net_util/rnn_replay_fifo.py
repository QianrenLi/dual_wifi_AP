from typing import List, Optional, Iterable, Dict
from net_util import register_buffer
import torch as th
import numpy as np
import random
import math

# ---------------- helpers (unchanged pieces) ----------------
def _as_1d_float(x):
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)

def _reward_agg_fn(reward_agg):
    if reward_agg == "sum":  return lambda arr: float(np.asarray(arr, np.float32).sum())
    if reward_agg == "mean": return lambda arr: float(np.asarray(arr, np.float32).mean())
    if callable(reward_agg): return lambda arr: float(reward_agg(np.asarray(arr, np.float32)))
    raise ValueError

def _tracify(states, actions, rewards, network_output, reward_agg):
    T = len(states); assert len(actions)==T and len(rewards)==T
    obs_np = np.stack([_as_1d_float(s) for s in states], axis=0).astype(np.float32)
    act_np = np.stack([_as_1d_float(a) for a in actions], axis=0).astype(np.float32)
    agg = _reward_agg_fn(reward_agg)
    rew_np = np.asarray([agg(_as_1d_float(r)) for r in rewards], dtype=np.float32)
    next_obs_np = np.vstack([obs_np[1:], np.zeros((1, obs_np.shape[1]), np.float32)]) if T>1 else np.zeros((1, obs_np.shape[1]), np.float32)
    done_np = np.array([float(network_output[t].get("done",0)) for t in range(T)], dtype=np.float32)
    done_np[-1] = 1.0
    return obs_np, act_np, rew_np, next_obs_np, done_np

def summarize(xs: np.ndarray):
    n = xs.size
    if n == 0: return (0,0.0,0.0)
    mean = float(xs.mean()); centered = xs - mean
    M2 = float(np.dot(centered, centered))
    return (n, mean, M2)

def merge(a, b):
    n_a, mean_a, M2_a = a; n_b, mean_b, M2_b = b
    if n_b==0: return a
    if n_a==0: return b
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    M2   = M2_a + M2_b + delta * delta * (n_a * n_b / n)
    return (n, mean, M2)

def var_unbiased(summary):
    n, _, M2 = summary
    return M2 / (n - 1) if n > 1 else float("nan")

# ---------------- episode ----------------
# ---------------- episode (Modified to use NumPy) ----------------
class Episode:
    __slots__ = ("obs_np","actions_np","rewards_np","next_obs_np","dones_np","loss",
                 "reward_summary","gamma_summary","avg_return","interference",
                 "data_num","_heap_idx")
                 
    def __init__(self, obs_np, act_np, rew_np, next_obs_np, done_np,
                 init_loss: float, gamma: float, interference=0):
        
        self.obs_np = obs_np
        self.actions_np = act_np
        self.rewards_np = rew_np
        self.next_obs_np = next_obs_np
        self.dones_np = done_np

        # 'loss' is kept only for API compatibility / logging; no priority usage.
        self.loss = float(init_loss)
        self.reward_summary = summarize(rew_np)
        self.gamma_summary  = summarize((1 - done_np) * gamma)
        self.data_num = self.reward_summary[0]
        
        # Calculate avg_return (variance-like measure of returns)
        G = 0.0; sq = 0.0
        for t in range(self.data_num - 1, -1, -1):
            G = float(rew_np[t]) + gamma * G * (1.0 - float(done_np[t]))
            sq += G * G
        self.avg_return = sq / max(1, self.data_num)
        
        self.interference = float(interference)
        self._heap_idx = -1  # kept for compatibility, unused in FIFO

    @property
    def length(self): 
        return self.data_num


@register_buffer
class RNNPriReplayFiFo:
    """
    FIFO episode replay buffer with uniform sampling (no prioritization).

    API-compatible with the previous rank-based PER version:
      - same class name
      - same methods: build_from_traces, extend, add_episode, update_episode_losses,
        _approx_rank_probs, _choose_episode_ids, _calc_is_weights, get_sequences, etc.
    """
    def __init__(self,
                 device: str = "cuda",
                 capacity: int = 10000,
                 gamma: float = 0.99,
                 alpha: float = 0.7,     # kept for API compatibility, unused
                 beta0: float = 0.4,     # kept for API compatibility, unused
                 rebalance_interval: int = 100,   # unused
                 writer=None,
                 episode_length: int = 600):
        self.device = th.device(device)
        self.capacity = int(capacity)
        self.gamma = float(gamma)

        # These are kept for compatibility but no longer affect behavior.
        self.alpha = float(alpha)
        self.beta  = float(beta0)
        self.rebalance_interval = int(rebalance_interval)

        self.writer = writer
        self.episode_length = episode_length

        # FIFO storage: treat this as a ring buffer.
        self.heap: List[Episode] = []   # still called 'heap' for minimal changes
        self._next_insert: int = 0      # index to overwrite when full

        # stats you tracked
        self.reward_summary = (0,0.0,0.0)
        self.gamma_summary  = (0,0.0,0.0)
        self.avg_return = 0.0
        self.run_return = 0.0
        self.data_num = 0
        self.sigma = 0.0

    # ---------------- internal helpers (FIFO instead of heap) ----------------
    def _push(self, ep: Episode):
        """
        FIFO insert:
          - if buffer not full: append
          - else: overwrite the oldest episode at _next_insert
        """
        if len(self.heap) < self.capacity:
            ep._heap_idx = len(self.heap)
            self.heap.append(ep)
        else:
            # Overwrite the oldest episode
            self.heap[self._next_insert] = ep
            ep._heap_idx = self._next_insert

        # advance circular pointer
        if self.capacity > 0:
            self._next_insert = (self._next_insert + 1) % self.capacity

    # Kept for API compatibility; now no-ops.
    def _rebalance_if_needed(self, just_updated: int = 1):
        return

    # ---------------- construction / add / extend ----------------
    @staticmethod
    def build_from_traces(traces, device="cuda", reward_agg="sum", capacity=10000,
                          init_loss: Optional[float] = None, episode_length=600, **kwargs):
        buf = RNNPriReplayFiFo(
            device=device, 
            capacity=capacity, 
            gamma=kwargs.get("gamma", 0.99),
            alpha=kwargs.get("alpha", 0.7),            # ignored
            beta0=kwargs.get("beta0", 0.4),            # ignored
            rebalance_interval=kwargs.get("rebalance_interval", 100),  # ignored
            writer=kwargs.get("writer", None),
            episode_length=episode_length
        )
        interfs = kwargs.get("interference_vals", [0]*len(traces))
        for (states, actions, rewards, network_output), interf in zip(traces, interfs):
            buf.add_episode(states, actions, rewards, network_output, reward_agg, init_loss, interf)
        return buf

    def extend(self, traces, reward_agg="sum", init_loss: Optional[float] = None, **kwargs):
        interfs = kwargs.get("interference_vals", [0]*len(traces))
        for (states, actions, rewards, network_output), interf in zip(traces, interfs):
            self.add_episode(states, actions, rewards, network_output, reward_agg, init_loss, interf)

    def add_episode(self, states, actions, rewards, network_output,
                    reward_agg="sum", init_loss: Optional[float] = None, interference=0):
        obs_np, act_np, rew_np, next_obs_np, done_np = _tracify(
            states, actions, rewards, network_output, reward_agg
        )
        
        # Decompose the trace into multiple episodes of the specified length
        num_episodes = len(obs_np) // self.episode_length
        for i in range(num_episodes):
            start = i * self.episode_length
            end = (i + 1) * self.episode_length
            
            ep = Episode(
                obs_np[start:end], 
                act_np[start:end], 
                rew_np[start:end], 
                next_obs_np[start:end], 
                done_np[start:end],
                init_loss=10000.0 if init_loss is None else float(init_loss),
                gamma=self.gamma,
                interference=interference
            )

            # Update running stats (unchanged)
            self.reward_summary = merge(self.reward_summary, ep.reward_summary)
            self.gamma_summary  = merge(self.gamma_summary, ep.gamma_summary)
            if self.reward_summary[0] > 0:
                w = ep.reward_summary[0] / self.reward_summary[0]
                self.avg_return += (ep.avg_return - self.avg_return) * w

            for r in ep.rewards_np.tolist():
                self.run_return = float(r) + self.gamma * self.run_return
                self.data_num += 1
                if self.writer is not None:
                    self.writer.add_scalar("data/return", self.run_return, self.data_num)

            self.sigma = math.sqrt(
                max(0.0, var_unbiased(self.reward_summary)) +
                max(0.0, var_unbiased(self.gamma_summary)) * max(0.0, self.avg_return)
            )

            # FIFO insert; eviction handled inside _push
            self._push(ep)

    def update_episode_losses(self, ep_ids: List[int], losses: Iterable[float]):
        """
        Keep API but only update the stored loss values; no priority, no reordering.
        """
        for eid, new_loss in zip(ep_ids, losses):
            if 0 <= eid < len(self.heap):
                self.heap[eid].loss = float(new_loss)
            else:
                raise IndexError(
                    f"Episode ID {eid} out of range for replay buffer of size {len(self.heap)}"
                )
        # no rebalance needed
        self._rebalance_if_needed(just_updated=len(ep_ids))

    # ---------------- sampling (uniform instead of rank-based PER) ----------------
    def _approx_rank_probs(self):
        """
        For FIFO/uniform replay, just return uniform probabilities over all episodes.
        Kept for API compatibility.
        """
        N = len(self.heap)
        if N == 0:
            return None
        probs = np.full(N, 1.0 / N, dtype=np.float64)
        return probs
    
    def _choose_episode_ids(self, batch_size: int):
        """
        Sample episodes uniformly at random (with replacement).
        Returns:
          ep_ids: list[int]
          probs:  np.ndarray of shape [N], the marginal prob of each episode (uniform).
        """
        N = len(self.heap)
        probs = self._approx_rank_probs()
        if probs is None:
            return [], None
        idxs = np.arange(N, dtype=np.int64)
        chosen = np.random.choice(idxs, size=batch_size, replace=True, p=probs)
        return chosen.tolist(), probs

    def _calc_is_weights(
        self,
        probs_ep: np.ndarray,
        beta: Optional[float] = None,
        device: Optional[th.device | str] = None,
    ) -> th.Tensor:
        """
        Importance-sampling weights for PER.
        For FIFO/uniform replay we just return all-ones weights (no correction needed),
        but keep the API identical.
        """
        if probs_ep is None or probs_ep.size == 0:
            return th.zeros((0, 1), dtype=th.float32, device=device or self.device)

        wi = np.ones_like(probs_ep, dtype=np.float32)
        w_t = th.from_numpy(wi)
        dev = th.device(device) if device is not None else self.device
        w_t = w_t.to(dev, non_blocking=True)
        return w_t.unsqueeze(-1)
    
    def _gather_batch(
        self,
        ep_ids: List[int],
        T: int,
        device: Optional[th.device | str] = None,
    ):
        """
        Directly assemble batch from episode arrays using np.stack.
        Assumes each episode has length >= T and same length.
        Returns tensors on `device`.
        """
        if not ep_ids:
            return None, None, None, None, None, None

        dev = th.device(device) if device is not None else self.device

        eps = [self.heap[eid] for eid in ep_ids]
        B = len(eps)

        obs_TB   = np.stack([ep.obs_np[:T]     for ep in eps], axis=1)            # (T, B, obs_dim)
        act_TB   = np.stack([ep.actions_np[:T] for ep in eps], axis=1)            # (T, B, act_dim)
        rew_TB   = np.stack([ep.rewards_np[:T] for ep in eps], axis=1)[..., None] # (T, B, 1)
        nxt_TB   = np.stack([ep.next_obs_np[:T] for ep in eps], axis=1)           # (T, B, obs_dim)
        done_TB  = np.stack([ep.dones_np[:T]    for ep in eps], axis=1)[..., None]# (T, B, 1)
        interfs  = np.array([ep.interference    for ep in eps], dtype=np.float32) # (B,)

        obs_TB_t   = th.from_numpy(obs_TB).to(dev, non_blocking=True)
        act_TB_t   = th.from_numpy(act_TB).to(dev, non_blocking=True)
        rew_TB_t   = th.from_numpy(rew_TB).to(dev, non_blocking=True)
        nxt_TB_t   = th.from_numpy(nxt_TB).to(dev, non_blocking=True)
        done_TB_t  = th.from_numpy(done_TB).to(dev, non_blocking=True)
        interf_B_t = th.from_numpy(interfs).to(dev, non_blocking=True).unsqueeze(-1)

        return obs_TB_t, act_TB_t, rew_TB_t, nxt_TB_t, done_TB_t, interf_B_t

    def get_sequences(
        self,
        batch_size: int,
        device: Optional[th.device | str] = None,
        *args,
        **kwargs,
    ):
        """
        Yield a single batch of sequences:
          (obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info)
        API and shapes are unchanged.
        """
        if not self.heap:
            return

        dev = th.device(device) if device is not None else self.device

        ep_ids, probs_all = self._choose_episode_ids(batch_size)
        if not ep_ids:
            return

        idxs = np.asarray(ep_ids, dtype=np.int64)
        probs_ep = probs_all[idxs]  # [B]

        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(
            ep_ids, self.episode_length, device=dev
        )

        is_w_B1 = self._calc_is_weights(probs_ep, device=dev)
        prob_B1 = th.from_numpy(probs_ep.astype(np.float32)).to(
            dev, non_blocking=True
        ).unsqueeze(-1)

        info = {
            "ep_ids": ep_ids,
            "interference": interf_B,
            "is_weights": is_w_B1,
            "probs": prob_B1,
        }
        yield (obs_TB, act_TB, rew_TB, nxt_TB, done_TB, info)
