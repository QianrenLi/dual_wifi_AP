from collections import deque
import heapq
from typing import List, Iterable, Optional, Dict
from net_util import register_buffer
import torch as th
import numpy as np
import random

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
    __slots__ = ("id", "obs", "actions", "rewards", "next_obs", "dones", "loss", "load_t", "reward_summary", "gamma_summary", "avg_return")
    def __init__(self, obs_np, act_np, rew_np, next_obs_np, done_np, device, init_loss: float = 1.0, gamma = 0.99):
        self.obs = th.tensor(obs_np, device=device)
        self.actions = th.tensor(act_np, device=device)
        self.rewards = th.tensor(rew_np, device=device)
        self.next_obs = th.tensor(next_obs_np, device=device)
        self.dones = th.tensor(done_np, device=device)
        self.loss = float(init_loss)
        self.load_t = 0
        
        self.reward_summary = summarize(rew_np)
        self.gamma_summary = summarize( (1 - done_np) * gamma )
        
        G = 0
        self.avg_return = 0
        for t in range(self.reward_summary[0] - 1, -1, -1):
            G = rew_np[t] + gamma * G * (1.0 - done_np[t])
            self.avg_return += G ** 2
        self.avg_return /= self.reward_summary[0]
        

    def __lt__(self, other):
        if not isinstance(other, Episode):
            return NotImplemented
        return self.loss < other.loss

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])
    
    def reset_cursor(self):
        self.load_t = 0

    def load(self):
        if self.load_t >= self.length:
            raise StopIteration
        t = self.load_t
        self.load_t += 1
        return (
            self.obs[t:t+1],
            self.actions[t:t+1],
            self.rewards[t:t+1],
            self.next_obs[t:t+1],
            self.dones[t:t+1],
        )

# -------- Minimal Two-Tier RNN Replay --------
@register_buffer
class RNNPriReplayBuffer2:
    def __init__(self, device: str = "cuda", recent_capacity: int = 10, top_capacity: int = 300, gamma = 0.99):
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
        **_: dict,
    ):
        assert len(traces) >= 1
        buf = RNNPriReplayBuffer2(device=device, recent_capacity=recent_capacity, top_capacity=top_capacity)
        for (states, actions, rewards, network_output) in traces:
            buf.add_episode(states, actions, rewards, network_output, reward_agg, init_loss)
        return buf
    
    def extend(self, traces, reward_agg: str = "sum", init_loss: Optional[float] = None, **kwargs: dict):
        for (states, actions, rewards, network_output) in traces:
            self.add_episode(states, actions, rewards, network_output, reward_agg, init_loss)
            
            
    def add_episode(
        self,
        states,
        actions,
        rewards,
        network_output,
        reward_agg: str = "sum",
        init_loss: Optional[float] = None,
    ):
        ep = Episode(
            *_tracify(states, actions, rewards, network_output, reward_agg),
            device=self.device,
            init_loss=100.0 if init_loss is None else float(init_loss),
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

                                
    def get_minibatches(self, batch_size: int):
        # choose recent (shuffle for fairness)
        n_recent_avail = len(self.recent_k)
        n_recent = min(n_recent_avail, batch_size)
        recent_ids = list(range(n_recent_avail))
        random.shuffle(recent_ids)
        recent_ids = recent_ids[:n_recent]

        # choose from heap with prob ~ loss
        need = batch_size - n_recent
        heap_ids = []
        if need > 0 and len(self.heap) > 0:
            w = np.asarray([max(float(ep.loss), 1e-12) for ep in self.heap], dtype=np.float64)
            s = float(w.sum())
            p = (w / s) if (s > 0.0 and np.isfinite(s)) else None
            base = len(self.recent_k)
            picked = np.random.choice(np.arange(base, base + len(self.heap)),
                                    size=min(need, len(self.heap)),
                                    replace=False, p=p)
            heap_ids = picked.tolist()

        cands = recent_ids + heap_ids
        for ep_id in cands:
            self._id_to_ep(ep_id).reset_cursor()
            
        while True:
            O, A, R, NO, D = [], [], [], [], []
            try:
                # if any ep raises StopIteration, abort whole computation immediately
                for ep_id in cands:
                    o, a, r, no, d = self._id_to_ep(ep_id).load()
                    O.append(o); A.append(a); R.append(r); NO.append(no); D.append(d)
            except Exception as e:
                break  # stop all trace computation

            # yield this synchronized time-step across all chosen episodes
            obs_b      = th.cat(O,  dim=0)
            act_b      = th.cat(A,  dim=0)
            next_obs_b = th.cat(NO, dim=0)
            rew_b      = th.cat(R,  dim=0).unsqueeze(-1)
            done_b     = th.cat(D,  dim=0).unsqueeze(-1)
            yield obs_b, act_b, rew_b, next_obs_b, done_b, {"ep_ids": cands}

