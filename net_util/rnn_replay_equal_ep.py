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
    if T > 0: done_np[-1] = 1.0
    
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

    def start_point(self, T: int) -> int:
        L = max(0, self.data_num - T)
        return 0 if L <= 0 else np.random.randint(0, L)

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
                 episode_length: int = 100):
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
                          init_loss: Optional[float] = None, episode_length=100, **kwargs):
        buf = RNNPriReplayEqualEp(
            device=device, capacity=capacity, gamma=kwargs.get("gamma", 0.99),
            alpha=kwargs.get("alpha", 0.7), beta0=kwargs.get("beta0", 0.4),
            rebalance_interval=kwargs.get("rebalance_interval", 2000),
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
                init_loss=100.0 if init_loss is None else float(init_loss),
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
    
    def _gather_batch(self, ep_ids: List[int], starts: List[int], T: int):
        """
        IMPROVEMENT: Efficiently gathers batch data using NumPy array pre-allocation 
        and slicing before a single PyTorch conversion.
        """
        if not ep_ids:
            return None, None, None, None, None, None
            
        B = len(ep_ids)
        ep0 = self.heap[ep_ids[0]]
        obs_shape = ep0.obs_np.shape[1]
        act_shape = ep0.actions_np.shape[1]

        # 1. Pre-allocate large NumPy arrays (T, B, dim)
        obs_TB = np.zeros((T, B, obs_shape), dtype=np.float32)
        act_TB = np.zeros((T, B, act_shape), dtype=np.float32)
        rew_TB = np.zeros((T, B, 1), dtype=np.float32)
        nxt_TB = np.zeros((T, B, obs_shape), dtype=np.float32)
        done_TB = np.zeros((T, B, 1), dtype=np.float32)
        interfs = np.zeros(B, dtype=np.float32)

        # 2. Loop and slice/copy data into pre-allocated arrays
        for i, (eid, s) in enumerate(zip(ep_ids, starts)):
            ep = self.heap[eid]
            e = s + T
            
            # Use direct indexing (NumPy slices)
            obs_TB[:, i] = ep.obs_np[s:e]
            act_TB[:, i] = ep.actions_np[s:e]
            # Reshape 1D reward/done arrays for 3D batch array
            rew_TB[:, i, 0] = ep.rewards_np[s:e]
            nxt_TB[:, i] = ep.next_obs_np[s:e]
            done_TB[:, i, 0] = ep.dones_np[s:e]
            interfs[i] = ep.interference

        # 3. Convert all arrays to PyTorch Tensors on the target device simultaneously
        obs_TB   = th.from_numpy(obs_TB).to(self.device)
        act_TB   = th.from_numpy(act_TB).to(self.device)
        rew_TB   = th.from_numpy(rew_TB).to(self.device)
        nxt_TB   = th.from_numpy(nxt_TB).to(self.device)
        done_TB  = th.from_numpy(done_TB).to(self.device)
        interf_B = th.from_numpy(interfs).to(self.device).unsqueeze(-1)
        
        return obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B

    def _calc_is_weights(self, ep_ids: List[int], probs: np.ndarray, beta: Optional[float] = None):
        if beta is None: beta = self.beta
        N = len(self.heap)
        p = probs[np.asarray(ep_ids, dtype=np.int64)]
        wi = np.power(N * np.maximum(1e-12, p), -float(beta))
        wi /= wi.max() if wi.size > 0 else 1.0
        return th.tensor(wi, dtype=th.float32).unsqueeze(-1)
    
    def get_minibatches(self, batch_size: int, trace_length: int = 100, device: Optional[th.device|str] = None):
        if not self.heap: return
        
        dev = th.device(device) if device is not None else self.device
        
        ep_ids, probs = self._choose_episode_ids(batch_size)
        
        # No start points needed since every episode is of the same length
        starts = [0] * len(ep_ids)
        
        # _gather_batch now returns tensors already on self.device (cuda or cpu)
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)

        is_w_B1 = self._calc_is_weights(ep_ids, probs).to(dev)
        prob_B1 = th.tensor(probs[np.asarray(ep_ids)], dtype=th.float32).unsqueeze(-1).to(dev)

        info = {"ep_ids": ep_ids, "interference": interf_B, "is_weights": is_w_B1, "probs": prob_B1}
        T = trace_length
        for t in range(T):
            yield (obs_TB[t], act_TB[t], rew_TB[t], nxt_TB[t], done_TB[t], info)
