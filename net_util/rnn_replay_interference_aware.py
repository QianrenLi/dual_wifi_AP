from typing import List, Optional, Iterable, Dict, Tuple, Any
from net_util import register_buffer
import torch as th
import numpy as np
import random
import math

# ---------------- helpers (unchanged) ----------------
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

# ---------------- episode (unchanged) ----------------
class Episode:
    __slots__ = ("obs","actions","rewards","next_obs","dones","loss",
                 "reward_summary","gamma_summary","avg_return","interference",
                 "data_num","_heap_idx")
    def __init__(self, obs_np, act_np, rew_np, next_obs_np, done_np, device,
                 init_loss: float, gamma: float, interference=0):
        self.obs = th.tensor(obs_np, device=device)
        self.actions = th.tensor(act_np, device=device)
        self.rewards = th.tensor(rew_np, device=device)
        self.next_obs = th.tensor(next_obs_np, device=device)
        self.dones = th.tensor(done_np, device=device)
        self.loss = float(init_loss)
        self.reward_summary = summarize(rew_np)
        self.gamma_summary  = summarize((1 - done_np) * gamma)
        self.data_num = self.reward_summary[0]
        G = 0.0; sq = 0.0
        for t in range(self.data_num-1, -1, -1):
            G = float(rew_np[t]) + gamma * G * (1.0 - float(done_np[t]))
            sq += G*G
        self.avg_return = sq / max(1, self.data_num)
        self.interference = float(interference)
        self._heap_idx = -1  # maintained by heap

    @property
    def length(self): return int(self.obs.shape[0])

    def start_point(self, T: int) -> int:
        L = max(0, self.data_num - T)
        return 0 if L <= 0 else np.random.randint(0, L)

# ---------------- single-heap utility used per interference ----------------
class _InterferenceHeap:
    """Array-based max-heap keyed by Episode.loss."""
    def __init__(self, rebalance_interval: int):
        self.heap: List[Episode] = []
        self._steps_since_rebalance = 0
        self.rebalance_interval = int(rebalance_interval)

    def __len__(self): return len(self.heap)

    def _key(self, ep: Episode) -> float: return ep.loss

    def _swap(self, i: int, j: int):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.heap[i]._heap_idx = i
        self.heap[j]._heap_idx = j

    def _sift_up(self, i: int):
        while i > 0:
            p = (i - 1) >> 1
            if self._key(self.heap[p]) >= self._key(self.heap[i]): break
            self._swap(i, p); i = p

    def _sift_down(self, i: int):
        n = len(self.heap)
        while True:
            l = (i << 1) + 1; r = l + 1
            largest = i
            if l < n and self._key(self.heap[l]) > self._key(self.heap[largest]): largest = l
            if r < n and self._key(self.heap[r]) > self._key(self.heap[largest]): largest = r
            if largest == i: break
            self._swap(i, largest); i = largest

    def push(self, ep: Episode):
        ep._heap_idx = len(self.heap)
        self.heap.append(ep)
        self._sift_up(ep._heap_idx)

    def remove_at(self, i: int):
        n = len(self.heap)
        if i < 0 or i >= n: return
        last = n - 1
        if i != last: self._swap(i, last)
        ep = self.heap.pop()
        ep._heap_idx = -1
        if i < len(self.heap):
            self._sift_down(i); self._sift_up(i)

    def update_key(self, idx: int, new_loss: float):
        if idx < 0 or idx >= len(self.heap): return
        ep = self.heap[idx]
        old = ep.loss; ep.loss = float(new_loss)
        if ep.loss > old: self._sift_up(idx)
        else: self._sift_down(idx)

    def evict_random_leaf(self):
        n = len(self.heap)
        if n == 0: return
        leaf_start = n >> 1
        idx = random.randint(leaf_start, n - 1)
        self.remove_at(idx)

    def rebalance_if_needed(self, just_updated: int = 1):
        self._steps_since_rebalance += max(1, int(just_updated))
        if self._steps_since_rebalance >= self.rebalance_interval and len(self.heap) > 1:
            self._steps_since_rebalance = 0
            self.heap.sort(key=lambda e: e.loss, reverse=True)
            for i, ep in enumerate(self.heap): ep._heap_idx = i
            for i in range((len(self.heap) >> 1) - 1, -1, -1):
                self._sift_down(i)

    # rank-based probs using heap array as approximate rank
    def approx_rank_probs(self, alpha: float):
        N = len(self.heap)
        if N == 0: return None
        ranks = (np.arange(N, dtype=np.int64) + 1).astype(np.float64)
        probs = 1.0 / np.power(ranks, max(0.0, alpha))
        probs /= probs.sum()
        return probs

# ---------------- multi-heap replay buffer ----------------
@register_buffer
class RNNPriReplayInterferenceAware:
    """
    Multi-heap, interference-aware prioritized replay.
    - One max-heap per distinct `interference` value.
    - Global capacity across all heaps; evict from the largest heap when over.
    - Two-stage sampling: choose heap ~ size/total; then rank-based sample within heap.
    - `ep_ids` returned are tuples: (interference_key, local_idx).
      Pass them unchanged to `update_episode_losses`.
      (Legacy plain-int IDs are still accepted and treated as single-heap indices.)
    """
    def __init__(self,
                 device: str = "cuda",
                 capacity: int = 10000,
                 gamma: float = 0.99,
                 alpha: float = 0.7,
                 beta0: float = 0.4,
                 rebalance_interval: int = 2000,
                 writer=None):
        self.device = device
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta  = float(beta0)
        self.rebalance_interval = int(rebalance_interval)
        self.writer = writer

        # interference_value -> _InterferenceHeap
        self.interference_heaps: Dict[float, _InterferenceHeap] = {}

        # stats (global, unchanged)
        self.reward_summary = (0,0.0,0.0)
        self.gamma_summary  = (0,0.0,0.0)
        self.avg_return = 0.0
        self.run_return = 0.0
        self.data_num = 0
        self.sigma = 0.0

    # --------- heap accessors ----------
    def _get_or_create_heap(self, interference: float) -> _InterferenceHeap:
        key = float(interference)
        if key not in self.interference_heaps:
            self.interference_heaps[key] = _InterferenceHeap(self.rebalance_interval)
        return self.interference_heaps[key]

    def _total_size(self) -> int:
        return sum(len(h) for h in self.interference_heaps.values())

    def _largest_heap_key(self) -> Optional[float]:
        if not self.interference_heaps: return None
        return max(self.interference_heaps.keys(), key=lambda k: len(self.interference_heaps[k]))

    # ---------------- construction / add / extend ----------------
    @staticmethod
    def build_from_traces(traces, device="cuda", reward_agg="sum", capacity=10000,
                          init_loss: Optional[float] = None, **kwargs):
        buf = RNNPriReplayInterferenceAware(
            device=device, capacity=capacity, gamma=kwargs.get("gamma", 0.99),
            alpha=kwargs.get("alpha", 0.7), beta0=kwargs.get("beta0", 0.4),
            rebalance_interval=kwargs.get("rebalance_interval", 2000),
            writer=kwargs.get("writer", None)
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
        obs_np, act_np, rew_np, next_obs_np, done_np = _tracify(states, actions, rewards, network_output, reward_agg)
        ep = Episode(
            obs_np, act_np, rew_np, next_obs_np, done_np,
            device=self.device,
            init_loss=100.0 if init_loss is None else float(init_loss),
            gamma=self.gamma,
            interference=interference
        )

        # global running stats
        self.reward_summary = merge(self.reward_summary, ep.reward_summary)
        self.gamma_summary  = merge(self.gamma_summary, ep.gamma_summary)
        if self.reward_summary[0] > 0:
            w = ep.reward_summary[0] / self.reward_summary[0]
            self.avg_return += (ep.avg_return - self.avg_return) * w
        for r in ep.rewards.detach().cpu().numpy().tolist():
            self.run_return = float(r) + self.gamma * self.run_return
            self.data_num += 1
            if self.writer is not None:
                self.writer.add_scalar("data/return", self.run_return, self.data_num)
        self.sigma = math.sqrt(
            max(0.0, var_unbiased(self.reward_summary)) +
            max(0.0, var_unbiased(self.gamma_summary)) * max(0.0, self.avg_return)
        )

        # insert into the appropriate heap
        heap = self._get_or_create_heap(ep.interference)
        heap.push(ep)

        # global capacity control: evict from the largest heap
        if self._total_size() > self.capacity:
            k = self._largest_heap_key()
            if k is not None:
                self.interference_heaps[k].evict_random_leaf()

        # light maintenance
        heap.rebalance_if_needed()

    # ---------------- priority updates ----------------
    def update_episode_losses(self, ep_ids: List[Any], losses: Iterable[float]):
        """
        Accepts IDs returned from sampling:
        - New format: (interference_key: float, local_idx: int)
        - Legacy single-heap int indices are also accepted (no-op here unless you migrated data).
        """
        for eid, new_loss in zip(ep_ids, losses):
            if isinstance(eid, tuple) and len(eid) == 2:
                k, local_idx = float(eid[0]), int(eid[1])
                heap = self.interference_heaps.get(k, None)
                if heap is not None:
                    heap.update_key(local_idx, float(new_loss))
                    heap.rebalance_if_needed(just_updated=1)
            else:
                # Legacy compatibility path: if a single heap happened to exist with key 0.0
                # and IDs were plain ints, update that one.
                heap = self.interference_heaps.get(0.0, None)
                if heap is not None and isinstance(eid, int):
                    heap.update_key(eid, float(new_loss))
                    heap.rebalance_if_needed(just_updated=1)

    # ---------------- sampling helpers ----------------
    def _choose_heap_keys(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays of chosen heap keys and their probabilities p(heap)."""
        keys = np.array(list(self.interference_heaps.keys()), dtype=np.float64)
        sizes = np.array([len(self.interference_heaps[k]) for k in self.interference_heaps], dtype=np.float64)
        total = sizes.sum()
        if total <= 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        p_heap = sizes / total
        chosen = np.random.choice(keys, size=size, replace=True, p=p_heap)
        return chosen, p_heap  # p_heap aligned with 'keys' order; we’ll reindex as needed

    def _heap_prob_lookup(self) -> Dict[float, float]:
        sizes = {k: float(len(h)) for k, h in self.interference_heaps.items()}
        total = sum(sizes.values())
        if total <= 0: return {k: 0.0 for k in self.interference_heaps}
        return {k: v/total for k, v in sizes.items()}

    def _choose_episode_ids(self, batch_size: int):
        """
        Two-stage:
        1) choose heap key ~ size/total
        2) within that heap, choose index ~ rank-based probs
        Returns:
          - ep_ids: list of (heap_key, local_idx)
          - probs: np.ndarray of combined probabilities p(heap)*p(local|heap)
        """
        if self._total_size() == 0:
            return [], None

        # Stage 1: heap sampling
        heap_keys = list(self.interference_heaps.keys())
        heap_p = self._heap_prob_lookup()
        if not heap_keys:
            return [], None

        chosen_heaps = np.random.choice(
            heap_keys, size=batch_size, replace=True,
            p=np.array([heap_p[k] for k in heap_keys], dtype=np.float64)
        )

        ep_ids: List[Tuple[float, int]] = []
        probs = np.zeros((batch_size,), dtype=np.float64)

        for i, hk in enumerate(chosen_heaps):
            heap = self.interference_heaps[hk]
            local_probs = heap.approx_rank_probs(self.alpha)
            # If this heap is empty (shouldn't happen with our chooser), fall back to any non-empty heap
            if local_probs is None or local_probs.size == 0:
                # try to find a non-empty heap
                non_empty = [k for k in heap_keys if len(self.interference_heaps[k]) > 0]
                if not non_empty:
                    continue
                hk = random.choice(non_empty)
                heap = self.interference_heaps[hk]
                local_probs = heap.approx_rank_probs(self.alpha)

            local_idx = int(np.random.choice(np.arange(len(heap.heap)), p=local_probs))
            ep_ids.append((float(hk), local_idx))
            probs[i] = float(heap_p[hk]) * float(local_probs[local_idx])

        return ep_ids, probs

    def _gather_batch(self, ep_ids: List[Tuple[float, int]], starts: List[int], T: int):
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB = [], [], [], [], []
        interfs = []
        for (hk, local_idx), s in zip(ep_ids, starts):
            heap = self.interference_heaps[float(hk)]
            ep = heap.heap[local_idx]
            e = s + T
            obs_TB.append(ep.obs[s:e]); act_TB.append(ep.actions[s:e])
            rew_TB.append(ep.rewards[s:e]); nxt_TB.append(ep.next_obs[s:e])
            done_TB.append(ep.dones[s:e]); interfs.append(ep.interference)

        obs_TB  = th.stack(obs_TB, 0).transpose(0,1).contiguous()
        act_TB  = th.stack(act_TB, 0).transpose(0,1).contiguous()
        rew_TB  = th.stack(rew_TB, 0).transpose(0,1).contiguous().unsqueeze(-1)
        nxt_TB  = th.stack(nxt_TB, 0).transpose(0,1).contiguous()
        done_TB = th.stack(done_TB,0).transpose(0,1).contiguous().unsqueeze(-1)
        interf_B = th.tensor(interfs, device=obs_TB.device, dtype=th.float32).unsqueeze(-1)
        return obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B

    def _calc_is_weights(self, probs: np.ndarray, beta: Optional[float] = None):
        if beta is None: beta = self.beta
        N = float(self._total_size())
        p = np.maximum(1e-12, probs.astype(np.float64))
        wi = np.power(N * p, -float(beta))
        wi /= wi.max() if wi.size > 0 else 1.0
        return th.tensor(wi, dtype=th.float32).unsqueeze(-1)

    # ---------------- public batch APIs ----------------
    def get_minibatches(self, batch_size: int, trace_length: int = 100, device: Optional[th.device|str] = None):
        if self._total_size() == 0: return
        dev = th.device(device) if device is not None else None

        ep_ids, probs = self._choose_episode_ids(batch_size)
        if not ep_ids: return

        starts = []
        # --- FIX: Adjust probability for trace/slice sampling ---
        # The probability of picking a specific slice is P(Episode) * (1 / Valid_Starts).
        # We must divide by valid_length to correct IS weights for long episodes.
        for i, (hk, local_idx) in enumerate(ep_ids):
            ep = self.interference_heaps[float(hk)].heap[int(local_idx)]
            valid_len = max(1, ep.data_num - trace_length)
            starts.append(ep.start_point(trace_length))
            probs[i] /= float(valid_len)
        # --------------------------------------------------------

        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)

        is_w_B1 = self._calc_is_weights(probs)
        prob_B1 = th.tensor(probs, dtype=th.float32).unsqueeze(-1)

        if dev is not None and dev.type == "cuda" and obs_TB.device.type == "cpu":
            obs_TB   = obs_TB.pin_memory().to(dev, non_blocking=True)
            act_TB   = act_TB.pin_memory().to(dev, non_blocking=True)
            rew_TB   = rew_TB.pin_memory().to(dev, non_blocking=True)
            nxt_TB   = nxt_TB.pin_memory().to(dev, non_blocking=True)
            done_TB  = done_TB.pin_memory().to(dev, non_blocking=True)
            interf_B = interf_B.pin_memory().to(dev, non_blocking=True)

        if dev is not None and dev.type == "cuda" and is_w_B1.device.type == "cpu":
            is_w_B1  = is_w_B1.pin_memory().to(dev, non_blocking=True)
            prob_B1  = prob_B1.pin_memory().to(dev, non_blocking=True)

        info = {"ep_ids": ep_ids, "interference": interf_B, "is_weights": is_w_B1, "probs": prob_B1}
        T = trace_length
        for t in range(T):
            yield (obs_TB[t], act_TB[t], rew_TB[t], nxt_TB[t], done_TB[t], info)

    def get_sequences(self, batch_size: int, trace_length: int = 100, device: Optional[th.device|str] = None):
        # keep your warmup condition if desired; here preserved
        if self._total_size() == 0 or self.data_num < 10e3: return
        dev = th.device(device) if device is not None else None

        ep_ids, probs = self._choose_episode_ids(batch_size)
        if not ep_ids: return

        starts = []
        # --- FIX: Adjust probability for trace/slice sampling ---
        for i, (hk, local_idx) in enumerate(ep_ids):
            ep = self.interference_heaps[float(hk)].heap[int(local_idx)]
            valid_len = max(1, ep.data_num - trace_length)
            starts.append(ep.start_point(trace_length))
            probs[i] /= float(valid_len)
        # --------------------------------------------------------

        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)
        is_w_B1 = self._calc_is_weights(probs)
        prob_B1 = th.tensor(probs, dtype=th.float32).unsqueeze(-1)

        if dev is not None and dev.type == "cuda" and obs_TB.device.type == "cpu":
            obs_TB   = obs_TB.pin_memory().to(dev, non_blocking=True)
            act_TB   = act_TB.pin_memory().to(dev, non_blocking=True)
            rew_TB   = rew_TB.pin_memory().to(dev, non_blocking=True)
            nxt_TB   = nxt_TB.pin_memory().to(dev, non_blocking=True)
            done_TB  = done_TB.pin_memory().to(dev, non_blocking=True)
            interf_B = interf_B.pin_memory().to(dev, non_blocking=True)

        if dev is not None and dev.type == "cuda" and is_w_B1.device.type == "cpu":
            is_w_B1  = is_w_B1.pin_memory().to(dev, non_blocking=True)
            prob_B1  = prob_B1.pin_memory().to(dev, non_blocking=True)

        info = {"ep_ids": ep_ids, "interference": interf_B, "is_weights": is_w_B1, "probs": prob_B1}
        yield (obs_TB, act_TB, rew_TB, nxt_TB, done_TB, info)