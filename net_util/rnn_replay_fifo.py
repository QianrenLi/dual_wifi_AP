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
        self.loss = float(init_loss)  # kept but unused for sampling
        self.reward_summary = summarize(rew_np)
        self.gamma_summary  = summarize((1 - done_np) * gamma)
        self.data_num = self.reward_summary[0]
        G = 0.0; sq = 0.0
        for t in range(self.data_num-1, -1, -1):
            G = float(rew_np[t]) + gamma * G * (1.0 - float(done_np[t]))
            sq += G*G
        self.avg_return = sq / max(1, self.data_num)
        self.interference = float(interference)
        self._heap_idx = -1  # maintained by buffer

    @property
    def length(self): return int(self.obs.shape[0])

    def start_point(self, T: int) -> int:
        L = max(0, self.data_num - T)
        return 0 if L <= 0 else np.random.randint(0, L)

# ---------------- uniform (no-PER) replay with array list ----------------
@register_buffer
class RNNPriReplayFiFo:
    """
    Uniform replay:
    - Episodes stored in a simple list.
    - Sampling is uniform over episodes (with replacement).
    - Importance-sampling weights are all ones.
    """
    def __init__(self,
                 device: str = "cuda",
                 capacity: int = 10000,
                 gamma: float = 0.99,
                 **kwargs):
        self.device = device
        self.capacity = int(capacity)
        self.gamma = float(gamma)

        self.heap: List[Episode] = []  # simple list; keep name for compatibility

        # stats (kept from original)
        self.reward_summary = (0,0.0,0.0)
        self.gamma_summary  = (0,0.0,0.0)
        self.avg_return = 0.0
        self.run_return = 0.0
        self.data_num = 0
        self.sigma = 0.0
        
        self.writer = kwargs.get('writer', None)

    # ------------- list utilities (names kept for compatibility) -------------
    def _swap(self, i: int, j: int):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.heap[i]._heap_idx = i
        self.heap[j]._heap_idx = j

    def _push(self, ep: Episode):
        ep._heap_idx = len(self.heap)
        self.heap.append(ep)

    def _remove_at(self, i: int):
        n = len(self.heap)
        if i < 0 or i >= n: return
        last = n - 1
        if i != last:
            self._swap(i, last)
        ep = self.heap.pop()
        ep._heap_idx = -1

    def _evict_random(self):
        n = len(self.heap)
        if n == 0: return
        idx = random.randint(0, n - 1)
        self._remove_at(idx)

    def _rebalance_if_needed(self, just_updated: int = 1):
        # No-op in uniform replay (kept for API compatibility)
        return

    # ---------------- construction / add / extend ----------------
    @staticmethod
    def build_from_traces(traces, device="cuda", reward_agg="sum", capacity=10000,
                          init_loss: Optional[float] = None, **kwargs):
        buf = RNNPriReplayFiFo(
            device=device, capacity=capacity, gamma=kwargs.get("gamma", 0.99),
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
            init_loss=100.0 if init_loss is None else float(init_loss),  # kept but unused
            gamma=self.gamma,
            interference=interference
        )

        # running stats (unchanged)
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

        # insert & capacity control
        self._push(ep)
        if len(self.heap) > self.capacity:
            self._evict_random()

    # ---------------- priority updates (no-op) ----------------
    def update_episode_losses(self, ep_ids: List[int], losses: Iterable[float]):
        # No prioritization -> nothing to update; keep API
        return

    # ---------------- sampling ----------------
    def _choose_episode_ids(self, batch_size: int):
        N = len(self.heap)
        if N == 0: return [], None
        # Uniform with replacement
        chosen = np.random.randint(0, N, size=batch_size, dtype=np.int64)
        # Uniform probs array of length N (for API compatibility)
        probs = np.full(N, 1.0 / N, dtype=np.float64)
        return chosen.tolist(), probs

    def _gather_batch(self, ep_ids: List[int], starts: List[int], T: int):
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB = [], [], [], [], []
        interfs = []
        for eid, s in zip(ep_ids, starts):
            ep = self.heap[eid]
            e = s + T
            obs_TB.append(ep.obs[s:e])
            act_TB.append(ep.actions[s:e])
            rew_TB.append(ep.rewards[s:e])
            nxt_TB.append(ep.next_obs[s:e])
            done_TB.append(ep.dones[s:e])
            interfs.append(ep.interference)
        obs_TB  = th.stack(obs_TB, 0).transpose(0,1).contiguous()
        act_TB  = th.stack(act_TB, 0).transpose(0,1).contiguous()
        rew_TB  = th.stack(rew_TB, 0).transpose(0,1).contiguous().unsqueeze(-1)
        nxt_TB  = th.stack(nxt_TB, 0).transpose(0,1).contiguous()
        done_TB = th.stack(done_TB,0).transpose(0,1).contiguous().unsqueeze(-1)
        interf_B = th.tensor(interfs, device=obs_TB.device, dtype=th.float32).unsqueeze(-1)
        return obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B

    def _calc_is_weights(self, ep_ids: List[int], probs: np.ndarray, beta: Optional[float] = None):
        # Always return ones (no importance correction in uniform sampling)
        B = len(ep_ids)
        return th.ones((B, 1), dtype=th.float32)

    # ---------------- public batch APIs ----------------
    def get_minibatches(self, batch_size: int, trace_length: int = 100, device: Optional[th.device|str] = None):
        if not self.heap: return
        dev = th.device(device) if device is not None else None
        ep_ids, probs = self._choose_episode_ids(batch_size)
        starts = [self.heap[eid].start_point(trace_length) for eid in ep_ids]
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)

        is_w_B1 = self._calc_is_weights(ep_ids, probs)
        # Keep "probs" tensor for compatibility (uniform 1/N)
        N = len(self.heap)
        prob_vals = np.full(len(ep_ids), 1.0 / max(1, N), dtype=np.float32)
        prob_B1 = th.tensor(prob_vals, dtype=th.float32).unsqueeze(-1)

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
        if not self.heap or self.data_num < 10e3: return
        dev = th.device(device) if device is not None else None
        ep_ids, probs = self._choose_episode_ids(batch_size)
        starts = [self.heap[eid].start_point(trace_length) for eid in ep_ids]
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)

        is_w_B1 = self._calc_is_weights(ep_ids, probs)
        N = len(self.heap)
        prob_vals = np.full(len(ep_ids), 1.0 / max(1, N), dtype=np.float32)
        prob_B1 = th.tensor(prob_vals, dtype=th.float32).unsqueeze(-1)

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
