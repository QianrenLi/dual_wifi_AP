from collections import deque
import heapq
from typing import List, Iterable, Optional, Dict
from net_util import register_buffer
import torch as th
import numpy as np
import random
import time

# -------- Helpers --------
def _as_1d_float(x):
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

def _tracify(states, actions, rewards, network_output, reward_agg):
    T = len(states)
    assert len(actions) == T and len(rewards) == T

    obs_np = np.stack([_as_1d_float(s) for s in states], axis=0).astype(np.float32)
    act_np = np.stack([_as_1d_float(a) for a in actions], axis=0).astype(np.float32)
    
    agg = _reward_agg_fn(reward_agg)
    rew_np = np.asarray([agg(_as_1d_float(r)) for r in rewards], dtype=np.float32)

    if T > 1:
        next_obs_np = np.vstack([obs_np[1:], np.zeros((1, obs_np.shape[1]), dtype=np.float32)])
    else:
        next_obs_np = np.zeros((1, obs_np.shape[1]), dtype=np.float32)

    done_np = np.array([float(network_output[t].get("done", 0)) for t in range(T)], dtype=np.float32)
    done_np[-1] = 1.0
    
    return obs_np, act_np, rew_np, next_obs_np, done_np

def merge(a, b):
    n_a, mean_a, M2_a = a
    n_b, mean_b, M2_b = b
    if n_b == 0: return a
    if n_a == 0: return b
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    M2 = M2_a + M2_b + delta * delta * (n_a * n_b / n)
    return (n, mean, M2)

def summarize(xs: np.ndarray):
    n = xs.size
    if n == 0:
        return (0, 0.0, 0.0)
    mean = float(xs.mean())
    centered = xs - mean
    M2 = float(np.dot(centered, centered))
    return (n, mean, M2)

def var_unbiased(summary):
    n, _, M2 = summary
    return M2 / (n - 1) if n > 1 else float("nan")

# -------- Episode --------
class Episode:
    __slots__ = ("id", "obs", "actions", "rewards", "next_obs", "dones", "loss", "load_t", "reward_summary", "gamma_summary", "avg_return", "interference", "data_num")
    def __init__(self, obs_np, act_np, rew_np, next_obs_np, done_np, device, init_loss: float = 1.0, gamma = 0.99, interference = 0):
        self.obs = th.tensor(obs_np, device=device)
        self.actions = th.tensor(act_np, device=device)
        self.rewards = th.tensor(rew_np, device=device)
        self.next_obs = th.tensor(next_obs_np, device=device)
        self.dones = th.tensor(done_np, device=device)
        self.loss = float(init_loss)
        
        self.reward_summary = summarize(rew_np)
        self.gamma_summary = summarize( (1 - done_np) * gamma )
        self.data_num = self.reward_summary[0]
        
        G = 0
        self.avg_return = 0
        for t in range(self.data_num - 1, -1, -1):
            G = rew_np[t] + gamma * G * (1.0 - done_np[t])
            self.avg_return += G ** 2
        self.avg_return /= self.data_num
        
        self.interference = interference
        

    def __lt__(self, other):
        if not isinstance(other, Episode):
            return NotImplemented
        return self.loss < other.loss

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])
    
    def start_point(self, trace_length):
        return np.random.randint(0, self.data_num - trace_length)

    def load(self, index):
        if index >= self.length:
            raise StopIteration
        t = index
        index += 1
        return (
            self.obs[t:t+1],
            self.actions[t:t+1],
            self.rewards[t:t+1],
            self.next_obs[t:t+1],
            self.dones[t:t+1],
            index,
        )

# -------- Minimal Two-Tier RNN Replay --------
@register_buffer
class RNNPriReplayFast:
    def __init__(self, device: str = "cuda", recent_capacity: int = 10, top_capacity: int = 10000, gamma = 0.99):
        self.device = device
        self.recent_k: List[Episode] = []
        self.heap: List[Episode] = []
        self.recent_capacity = recent_capacity
        self.top_capacity = top_capacity
        self.gamma = gamma
        
        self.reward_summary = (0,0,0)
        self.gamma_summary = (0,0,0)
        self.avg_return = 0
        
    @staticmethod
    def build_from_traces(
        traces,
        device: str = "cuda",
        reward_agg: str = "sum",
        recent_capacity: int = 10,
        top_capacity: int = 300,
        init_loss: Optional[float] = None,
        **kwargs: dict,
    ):
        assert len(traces) >= 1
        buf = RNNPriReplayFast(device=device, recent_capacity=recent_capacity, top_capacity=top_capacity)
        for (states, actions, rewards, network_output), interference in zip(traces, kwargs.get('interference_vals')):
            buf.add_episode(states, actions, rewards, network_output, reward_agg, init_loss, interference)
        return buf
    
    def extend(self, traces, reward_agg: str = "sum", init_loss: Optional[float] = None, **kwargs: dict):
        for (states, actions, rewards, network_output), interference in zip(traces, kwargs.get('interference_vals')):
            self.add_episode(states, actions, rewards, network_output, reward_agg, init_loss, interference)
            
            
    def add_episode(
        self,
        states,
        actions,
        rewards,
        network_output,
        reward_agg: str = "sum",
        init_loss: Optional[float] = None,
        interference = 0,
    ):
        ep = Episode(
            *_tracify(states, actions, rewards, network_output, reward_agg),
            device=self.device,
            init_loss=100.0 if init_loss is None else float(init_loss),
            interference = interference
        )
        self.reward_summary = merge(self.reward_summary, ep.reward_summary)
        self.gamma_summary = merge(self.gamma_summary, ep.gamma_summary) 
        self.avg_return += (ep.avg_return - self.avg_return) * ( ep.reward_summary[0] / self.reward_summary[0] )
        
        self.recent_k.append(ep)
        if len(self.recent_k) > self.recent_capacity:
            heapq.heappush(self.heap, self.recent_k.pop(0))
            if len(self.heap) > self.top_capacity:
                heapq.heappop(self.heap)

        self.sigma = np.sqrt( var_unbiased(self.reward_summary) + var_unbiased(self.gamma_summary) * self.avg_return )
        
    def _id_to_ep(self, i) -> Episode:
        split = len(self.recent_k)
        if i < split:
            return self.recent_k[i]
        else:
            return self.heap[i - split]
                
    def update_episode_losses(self, ep_ids, losses):
        split = len(self.recent_k)
        for eid, new_loss in zip(ep_ids, losses):
            if eid < split:
                self.recent_k[eid].loss = float(new_loss)
            else:
                self.heap[eid - split].loss = float(new_loss)
                
    def _choose_episode_ids(self, batch_size: int) -> List[int]:
        # recent
        n_recent_avail = len(self.recent_k)
        n_recent = min(n_recent_avail, batch_size)
        recent_ids = list(range(n_recent_avail))
        random.shuffle(recent_ids)
        recent_ids = recent_ids[:n_recent]

        # heap with prob ~ loss
        need = batch_size - n_recent
        heap_ids = []
        if need > 0 and len(self.heap) > 0:
            w = np.asarray([max(float(ep.loss), 1e-12) for ep in self.heap], dtype=np.float64)
            s = float(w.sum())
            p = (w / s) if (s > 0.0 and np.isfinite(s)) else None
            base = len(self.recent_k)
            picked = np.random.choice(np.arange(base, base + len(self.heap)), size=need, replace=True, p=p)
            heap_ids = picked.tolist()

        return recent_ids + heap_ids

    def _gather_batch(self, cands: List[int], starts: List[int], T: int):
        # Slice *once* per episode → stack to [T, B, ...] on CPU (buffer device)
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB = [], [], [], [], []
        interfs = []

        for ep_id, s in zip(cands, starts):
            ep = self._id_to_ep(ep_id)
            e = s + T
            # direct slicing; guaranteed valid by start_point()
            o  = ep.obs[s:e]          # [T, D]
            a  = ep.actions[s:e]      # [T, A]
            r  = ep.rewards[s:e]      # [T]
            no = ep.next_obs[s:e]     # [T, D]
            d  = ep.dones[s:e]        # [T]

            obs_TB.append(o)
            act_TB.append(a)
            rew_TB.append(r)
            nxt_TB.append(no)
            done_TB.append(d)
            interfs.append(ep.interference)

        # Stack along new batch dim → [T, B, ...]
        # (stack list of [T, *] → [B, T, *] then transpose for cache-friendly time-major)
        obs_TB  = th.stack(obs_TB,  dim=0).transpose(0, 1).contiguous()
        act_TB  = th.stack(act_TB,  dim=0).transpose(0, 1).contiguous()
        rew_TB  = th.stack(rew_TB,  dim=0).transpose(0, 1).contiguous().unsqueeze(-1)
        nxt_TB  = th.stack(nxt_TB,  dim=0).transpose(0, 1).contiguous()
        done_TB = th.stack(done_TB, dim=0).transpose(0, 1).contiguous().unsqueeze(-1)

        interf_B = th.tensor(interfs, device=obs_TB.device, dtype=th.float32).unsqueeze(-1)  # [B,1]
        return obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B
                                
    def get_minibatches(self, batch_size: int, trace_length: int = 100,
                    device: Optional[th.device | str] = None):
        """
        If `device` is CUDA: stack once on CPU, pin, do ONE non_blocking() copy to GPU,
        then yield GPU views [B, ...] per time step (zero-copy slicing).
        If `device` is CPU or None: identical behavior to before, but still zero-copy slicing.
        """
        if (len(self.recent_k) + len(self.heap)) == 0:
            return

        dev = th.device(device) if device is not None else None

        # --- choose episodes & starts once (unchanged) ---
        cands  = self._choose_episode_ids(batch_size)
        starts = [self._id_to_ep(eid).start_point(trace_length) for eid in cands]

        # --- gather a single time-major block on CPU (unchanged) ---
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(cands, starts, trace_length)

        # --- FAST PATH: one bulk, async copy to GPU then slice there ---
        if dev is not None and dev.type == "cuda":
            # pin then async copy the WHOLE block once
            obs_TB  = obs_TB.pin_memory().to(dev, non_blocking=True)
            act_TB  = act_TB.pin_memory().to(dev, non_blocking=True)
            rew_TB  = rew_TB.pin_memory().to(dev, non_blocking=True)
            nxt_TB  = nxt_TB.pin_memory().to(dev, non_blocking=True)
            done_TB = done_TB.pin_memory().to(dev, non_blocking=True)
            interf_B = interf_B.pin_memory().to(dev, non_blocking=True)

        info = {"ep_ids": cands, "interference": interf_B}  # [B,1] on CPU or GPU matching above
        T = trace_length
        for t in range(T):
            # These are *views* on CPU or GPU — no copies here.
            yield (obs_TB[t], act_TB[t], rew_TB[t], nxt_TB[t], done_TB[t], info)

    # in RNNPriReplayFast
    def get_sequences(self, batch_size: int, trace_length: int = 100,
                    device: Optional[th.device | str] = None):
        """
        Returns one full trace batch:
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, info
        Shapes: [T, B, ...] (time-major); info["ep_ids"] = list of B ints,
                info["interference"] = [B,1] tensor on same device.
        """
        if (len(self.recent_k) + len(self.heap)) == 0:
            return

        # sample episodes & starts once
        cands  = self._choose_episode_ids(batch_size)
        starts = [self._id_to_ep(eid).start_point(trace_length) for eid in cands]
        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B = self._gather_batch(cands, starts, trace_length)

        info = {"ep_ids": cands, "interference": interf_B}
        yield (obs_TB, act_TB, rew_TB, nxt_TB, done_TB, info)