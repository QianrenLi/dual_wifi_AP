from typing import List, Optional, Iterable, Dict, Tuple
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

# ---------------- episode ----------------
class Episode:
    __slots__ = ("obs","actions","rewards","next_obs","dones","loss",
                 "reward_summary","gamma_summary","avg_return","interference",
                 "data_num")
    def __init__(self, obs_np, act_np, rew_np, next_obs_np, done_np, device,
                 init_loss: float, gamma: float, interference=0):
        self.obs = th.tensor(obs_np, device=device)
        self.actions = th.tensor(act_np, device=device)
        self.rewards = th.tensor(rew_np, device=device)
        self.next_obs = th.tensor(next_obs_np, device=device)
        self.dones = th.tensor(done_np, device=device)
        self.loss = float(init_loss)  # kept for iface parity; not used by uniform buffer
        self.reward_summary = summarize(rew_np)
        self.gamma_summary  = summarize((1 - done_np) * gamma)
        self.data_num = self.reward_summary[0]
        G = 0.0; sq = 0.0
        for t in range(self.data_num-1, -1, -1):
            G = float(rew_np[t]) + gamma * G * (1.0 - float(done_np[t]))
            sq += G*G
        self.avg_return = sq / max(1, self.data_num)
        self.interference = float(interference)

    @property
    def length(self): return int(self.obs.shape[0])

    def start_point(self, T: int) -> int:
        L = max(0, self.data_num - T)
        return 0 if L <= 0 else np.random.randint(0, L)

# ---------------- uniform FIFO replay with circular storage ----------------
@register_buffer
class RNNUniformReplayFIFO:
    """
    FIFO episode buffer with UNIFORM sampling.

    - Episodes are stored in a fixed-size circular array.
    - When capacity is exceeded, the oldest episode is evicted automatically.
    - Sampling is uniform over current episodes; start points are uniform per episode.
    - Importance-sampling weights are all 1 (no PER).
    - Public API mirrors the previous buffer for easy swapping.
    """
    def __init__(self,
                 device: str = "cuda",
                 capacity: int = 10000,
                 gamma: float = 0.99,
                 writer=None):
        self.device   = device
        self.capacity = int(capacity)
        self.gamma    = float(gamma)
        self.writer   = writer

        # circular storage
        self._store: List[Optional[Episode]] = [None] * self.capacity
        self._head: int  = 0   # logical index of the oldest element
        self._size: int  = 0   # number of valid elements in store

        # running stats (cumulative, not a sliding-window)
        self.reward_summary = (0,0.0,0.0)
        self.gamma_summary  = (0,0.0,0.0)
        self.avg_return = 0.0
        self.run_return = 0.0
        self.data_num = 0
        self.sigma = 0.0

    # ----- circular helpers -----
    def _phys(self, logical_idx: int) -> int:
        """Map logical 0.._size-1 to physical index in the ring."""
        return (self._head + logical_idx) % self.capacity

    def _append_episode(self, ep: Episode):
        if self._size < self.capacity:
            pos = self._phys(self._size)
            self._store[pos] = ep
            self._size += 1
        else:
            # overwrite the oldest (FIFO)
            pos = self._phys(0)
            self._store[pos] = ep
            self._head = (self._head + 1) % self.capacity  # move head forward

    def _get_ep_by_logical(self, logical_idx: int) -> Episode:
        return self._store[self._phys(logical_idx)]  # type: ignore

    # ---------------- construction / add / extend ----------------
    @staticmethod
    def build_from_traces(traces, device="cuda", reward_agg="sum", capacity=10000,
                          init_loss: Optional[float] = None, **kwargs):
        buf = RNNUniformReplayFIFO(
            device=device, capacity=capacity, gamma=kwargs.get("gamma", 0.99),
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

        # cumulative (not windowed) stats
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

        # FIFO append
        self._append_episode(ep)

    # ---------------- priority updates (no-op for uniform FIFO) ----------------
    def update_episode_losses(self, ep_ids: List[int], losses: Iterable[float]):
        # uniform buffer does not use priorities; keep method for API parity
        return

    # ---------------- sampling (uniform) ----------------
    def _choose_episode_ids(self, batch_size: int) -> Tuple[List[int], np.ndarray]:
        N = self._size
        if N == 0:
            return [], np.empty((0,), dtype=np.float32)
        # logical ids 0..N-1
        chosen = np.random.randint(0, N, size=batch_size, dtype=np.int64)
        probs = np.full((N,), 1.0 / N, dtype=np.float64)  # uniform over episodes
        return chosen.tolist(), probs

    def _gather_batch(self, logical_ids: List[int], starts: List[int], T: int):
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB = [], [], [], [], []
        interfs = []
        for lid, s in zip(logical_ids, starts):
            ep = self._get_ep_by_logical(lid)
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

    def _ones_is_weights(self, n: int) -> th.Tensor:
        return th.ones((n, 1), dtype=th.float32, device=self.device)

    # ---------------- public batch APIs ----------------
    def get_minibatches(self, batch_size: int, trace_length: int = 100, device: Optional[th.device|str] = None):
        if self._size == 0: return
        dev = th.device(device) if device is not None else None

        ep_ids, probs = self._choose_episode_ids(batch_size)
        if len(ep_ids) == 0: return
        starts = [self._get_ep_by_logical(lid).start_point(trace_length) for lid in ep_ids]
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)

        # IS weights = 1; probs (for logging) = 1/N for chosen ids
        is_w_B1 = self._ones_is_weights(len(ep_ids))
        prob_B1 = th.full((len(ep_ids), 1), 1.0 / max(1, self._size), dtype=th.float32)

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
        if self._size == 0 or self.data_num < 10e3: return
        dev = th.device(device) if device is not None else None

        ep_ids, probs = self._choose_episode_ids(batch_size)
        if len(ep_ids) == 0: return
        starts = [self._get_ep_by_logical(lid).start_point(trace_length) for lid in ep_ids]
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(ep_ids, starts, trace_length)

        is_w_B1 = self._ones_is_weights(len(ep_ids))
        prob_B1 = th.full((len(ep_ids), 1), 1.0 / max(1, self._size), dtype=th.float32)

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
