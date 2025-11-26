from typing import List, Optional, Iterable, Dict
from net_util import register_buffer  # Assume this is available
import torch as th
import numpy as np
import random
import math

# ... summarize, merge, var_unbiased helpers remain unchanged ...
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

# ---------------- helpers (mostly unchanged pieces condensed) ----------------
def _as_1d_float(x):
    """Ensure input is a 1D float32 numpy array."""
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
    """Converts episode data lists into stacked NumPy arrays."""
    T = len(states); assert len(actions)==T and len(rewards)==T

    if isinstance(states[0], th.Tensor):
        obs_tensor = th.stack(states).detach().cpu()
        act_tensor = th.stack(actions).detach().cpu()
        obs_np = obs_tensor.numpy().astype(np.float32)
        act_np = act_tensor.numpy().astype(np.float32)
    else:
        obs_np = np.stack([_as_1d_float(s) for s in states], axis=0).astype(np.float32)
        act_np = np.stack([_as_1d_float(a) for a in actions], axis=0).astype(np.float32)

    agg = _reward_agg_fn(reward_agg)
    rew_np = np.asarray([agg(_as_1d_float(r)) for r in rewards], dtype=np.float32)
    
    # Handle next_obs/done array creation
    next_obs_np = np.vstack([obs_np[1:], np.zeros((1, obs_np.shape[1]), np.float32)]) if T>0 else np.zeros((1, obs_np.shape[1]), np.float32)
    done_np = np.array([float(network_output[t].get("done",0)) for t in range(T)], dtype=np.float32)
    # if T > 0: done_np[-1] = 1.0
    
    return obs_np, act_np, rew_np, next_obs_np, done_np

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

        self.loss = float(init_loss)
        self.reward_summary = summarize(rew_np)
        self.gamma_summary  = summarize((1 - done_np) * gamma)
        self.data_num = self.reward_summary[0]
        
        # Calculate avg_return (variance of returns)
        G = 0.0; sq = 0.0
        for t in range(self.data_num-1, -1, -1):
            G = float(rew_np[t]) + gamma * G * (1.0 - float(done_np[t]))
            sq += G*G
        self.avg_return = sq / max(1, self.data_num)
        
        self.interference = float(interference)
        self._heap_idx = -1  # maintained by buffer

    @property
    def length(self): return self.data_num

# ---------------- rank-based PER with array-based binary heap ----------------
@register_buffer
class RNNPriReplayEqualEp:
    """
    Array-based max-heap keyed by priority (episode.loss).
    - Sampling uses heap array index as an approximate rank (no full sort).
    - Periodic full sort & rebuild to keep the heap “not too unbalanced”, per paper.
    - Evict a random leaf when at capacity (approx low-priority).
    """
    def __init__(self,
                 device: str = "cuda",
                 capacity: int = 10000,
                 gamma: float = 0.99,
                 alpha: float = 0.7,     # rank-prioritization exponent
                 beta0: float = 0.4,     # IS exponent (you can anneal toward 1.0)
                 rebalance_interval: int = 2000,   # infrequent sort+rebuild
                 writer=None,
                 episode_length: int = 300):
        self.device = th.device(device)
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta  = float(beta0)
        self.rebalance_interval = int(rebalance_interval)
        self.writer = writer
        self.episode_length = episode_length  # Added episode length
        
        self.heap: List[Episode] = []       # max-heap on .loss
        self._steps_since_rebalance = 0

        # stats you tracked
        self.reward_summary = (0,0.0,0.0)
        self.gamma_summary  = (0,0.0,0.0)
        self.avg_return = 0.0
        self.run_return = 0.0
        self.data_num = 0
        self.sigma = 0.0

    # ... heap utilities (_key, _swap, _sift_up, _sift_down, _push, _remove_at, _update_key, _evict_leaf, _rebalance_if_needed) remain largely unchanged ...
    def _key(self, ep: Episode) -> float:
        return ep.loss

    def _swap(self, i: int, j: int):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.heap[i]._heap_idx = i
        self.heap[j]._heap_idx = j

    def _sift_up(self, i: int):
        while i > 0:
            p = (i - 1) >> 1
            if self._key(self.heap[p]) >= self._key(self.heap[i]): break
            self._swap(i, p)
            i = p

    def _sift_down(self, i: int):
        n = len(self.heap)
        while True:
            l = (i << 1) + 1
            r = l + 1
            largest = i
            if l < n and self._key(self.heap[l]) > self._key(self.heap[largest]):
                largest = l
            if r < n and self._key(self.heap[r]) > self._key(self.heap[largest]):
                largest = r
            if largest == i: break
            self._swap(i, largest)
            i = largest

    def _push(self, ep: Episode):
        ep._heap_idx = len(self.heap)
        self.heap.append(ep)
        self._sift_up(ep._heap_idx)

    def _remove_at(self, i: int):
        n = len(self.heap)
        if i < 0 or i >= n: return
        last = n - 1
        if i != last:
            self._swap(i, last)
        ep = self.heap.pop()
        ep._heap_idx = -1
        if i < len(self.heap):
            self._sift_down(i); self._sift_up(i)

    def _update_key(self, ep_idx: int, new_loss: float):
        if ep_idx < 0 or ep_idx >= len(self.heap): return
        ep = self.heap[ep_idx]
        old = ep.loss
        ep.loss = float(new_loss)
        if ep.loss > old:
            self._sift_up(ep_idx)
        else:
            self._sift_down(ep_idx)

    def _evict_leaf(self):
        n = len(self.heap)
        if n == 0: return
        leaf_start = n >> 1
        idx = random.randint(leaf_start, n - 1)
        self._remove_at(idx)

    def _rebalance_if_needed(self, just_updated: int = 1):
        self._steps_since_rebalance += max(1, int(just_updated))
        if self._steps_since_rebalance >= self.rebalance_interval and len(self.heap) > 1:
            self._steps_since_rebalance = 0
            self.heap.sort(key=lambda e: e.loss, reverse=True)
            for i, ep in enumerate(self.heap):
                ep._heap_idx = i
            for i in range((len(self.heap) >> 1) - 1, -1, -1):
                self._sift_down(i)
                
    # ---------------- construction / add / extend ----------------
    @staticmethod
    def build_from_traces(traces, device="cuda", reward_agg="sum", capacity=10000,
                          init_loss: Optional[float] = None, episode_length=600, **kwargs):
        buf = RNNPriReplayEqualEp(
            device=device, capacity=capacity, gamma=kwargs.get("gamma", 0.99),
            alpha=kwargs.get("alpha", 0.7), beta0=kwargs.get("beta0", 0.4),
            rebalance_interval=kwargs.get("rebalance_interval", 100),
            writer=kwargs.get("writer", None),
            episode_length=episode_length  # Passing episode_length
        )
        interfs = kwargs.get("interference_vals", [0]*len(traces))
        for (states, actions, rewards, network_output), interf in zip(traces, interfs):
            buf.add_episode(states, actions, rewards, network_output, reward_agg, init_loss, interf)
        return buf

    def extend(self, traces, reward_agg="sum", init_loss: Optional[float] = None, episode_length=100, **kwargs):
        interfs = kwargs.get("interference_vals", [0]*len(traces))
        for (states, actions, rewards, network_output), interf in zip(traces, interfs):
            self.add_episode(states, actions, rewards, network_output, reward_agg, init_loss, interf)

    def add_episode(self, states, actions, rewards, network_output,
                    reward_agg="sum", init_loss: Optional[float] = None, interference=0):
        obs_np, act_np, rew_np, next_obs_np, done_np = _tracify(states, actions, rewards, network_output, reward_agg)
        
        # Decompose the trace into multiple episodes of the specified length
        num_episodes = len(obs_np) // self.episode_length
        for i in range(num_episodes):
            start = i * self.episode_length
            end = (i + 1) * self.episode_length
            
            ep = Episode(
                obs_np[start:end], act_np[start:end], rew_np[start:end], next_obs_np[start:end], done_np[start:end],
                init_loss=10000.0 if init_loss is None else float(init_loss),
                gamma=self.gamma,
                interference=interference
            )

            # Update running stats
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

            # heap insert and capacity control
            self._push(ep)
            if len(self.heap) > self.capacity:
                self._evict_leaf()
            self._rebalance_if_needed()

    def update_episode_losses(self, ep_ids: List[int], losses: Iterable[float]):
        for eid, new_loss in zip(ep_ids, losses):
            if 0 <= eid < len(self.heap):
                self._update_key(eid, float(new_loss))
            else:
                raise IndexError(f"Episode ID {eid} out of range for replay buffer of size {len(self.heap)}")
        self._rebalance_if_needed(just_updated=len(ep_ids))
    
    # ---------------- sampling ----------------
    def _approx_rank_probs(self):
        N = len(self.heap)
        if N == 0: return None
        idxs = np.arange(N, dtype=np.int64)
        ranks = idxs + 1
        probs = 1.0 / np.power(ranks.astype(np.float64), max(0.0, self.alpha))
        probs /= probs.sum()
        return probs
    
    def _choose_episode_ids(self, batch_size: int):
        N = len(self.heap)
        probs = self._approx_rank_probs()
        if probs is None: return [], None
        chosen = np.random.choice(np.arange(N), size=batch_size, replace=True, p=probs)
        return chosen.tolist(), probs

    def _calc_is_weights(
        self,
        probs_ep: np.ndarray,
        beta: Optional[float] = None,
        device: Optional[th.device | str] = None,
    ) -> th.Tensor:
        """
        Compute importance-sampling weights from the per-sampled-episode probs.
        probs_ep: shape [B], probabilities of the sampled episodes.
        """
        if beta is None:
            beta = self.beta
        N = len(self.heap)
        p = np.maximum(1e-12, probs_ep)
        wi = np.power(N * p, -float(beta))
        wi /= wi.max() if wi.size > 0 else 1.0
        w_t = th.from_numpy(wi.astype(np.float32))
        if device is not None:
            w_t = w_t.to(device, non_blocking=True)
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

        # Grab all episodes once
        eps = [self.heap[eid] for eid in ep_ids]
        B = len(eps)

        # ---- Direct stacking along batch axis (axis=1 gives [T, B, ·]) ----
        obs_TB   = np.stack([ep.obs_np[:T]     for ep in eps], axis=1)            # (T, B, obs_dim)
        act_TB   = np.stack([ep.actions_np[:T] for ep in eps], axis=1)            # (T, B, act_dim)
        rew_TB   = np.stack([ep.rewards_np[:T] for ep in eps], axis=1)[..., None] # (T, B, 1)
        nxt_TB   = np.stack([ep.next_obs_np[:T] for ep in eps], axis=1)           # (T, B, obs_dim)
        done_TB  = np.stack([ep.dones_np[:T]    for ep in eps], axis=1)[..., None]# (T, B, 1)
        interfs  = np.array([ep.interference    for ep in eps], dtype=np.float32) # (B,)

        # ---- Single NumPy -> Torch hop per array ----
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

        
    
    # You can use the below method to sample from the replay buffer
    def sample(self, batch_size: int):
        """Sample a batch of episodes with uniform priority sampling."""
        ep_ids, probs = self._choose_episode_ids(batch_size)
        starts = [0] * len(ep_ids)  # No need for start points in this version
        return self._gather_batch(ep_ids, starts, self.episode_length)
