from dataclasses import dataclass
from typing import List, Optional, Dict, Iterable, Tuple
from collections import deque
import heapq
from net_util import register_buffer
import torch as th
import numpy as np
import random

# -----------------------------
# Helper Functions (unchanged)
# -----------------------------
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

# -----------------------------
# Episode (unchanged API)
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
        """Return one (o, a, r, no, d) at current cursor and advance; each with time axis length 1 (later we add [B,1,...])."""
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
# Two-tier RNN Replay Buffer
# -----------------------------
@register_buffer
@dataclass
class RNNPriReplayBuffer2:
    # device & ids
    device: str
    _next_eid: int = 0

    # two-tier storage
    recent_k: deque = None                     # deque[Episode], maxlen = recent_capacity
    recent_capacity: int = 10
    top_heap: list = None                      # list of (loss, eid) as a min-heap
    top_ids: set = None                        # set of eid in heap for O(1) membership
    top_capacity: int = 300
    episodes: Dict[int, Episode] = None        # eid -> Episode

    # internal flags
    _heap_dirty: bool = False                  # rebuild heap before sampling if True

    # ---------------- Construction ----------------
    @staticmethod
    def create(
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
        recent_capacity: int = 10,
        top_capacity: int = 300,
    ):
        return RNNPriReplayBuffer2(
            device=device,
            _next_eid=0,
            recent_k=deque(maxlen=recent_capacity),
            recent_capacity=recent_capacity,
            top_heap=[],
            top_ids=set(),
            top_capacity=top_capacity,
            episodes={},
            _heap_dirty=False,
        )

    # ---------------- Internals ----------------
    def _alloc_eid(self) -> int:
        eid = self._next_eid
        self._next_eid += 1
        return eid

    def _promote_to_heap(self, ep: Episode):
        """Insert an episode into the top-N heap by loss; evict lowest loss if at capacity."""
        if ep.id in self.top_ids:
            # already tracked; ensure heap gets refreshed on next sampling
            self._heap_dirty = True
            return
        if len(self.top_ids) < self.top_capacity:
            heapq.heappush(self.top_heap, (float(ep.loss), int(ep.id)))
            self.top_ids.add(int(ep.id))
        else:
            # compare with current smallest
            if self.top_heap and float(ep.loss) > float(self.top_heap[0][0]):
                popped_loss, popped_id = heapq.heapreplace(self.top_heap, (float(ep.loss), int(ep.id)))
                self.top_ids.discard(int(popped_id))
                self.top_ids.add(int(ep.id))
            # else: discard ep from top-N
        # no duplicates; heap may be slightly stale on loss updates
        # mark dirty only if we didn't just push (loss tracked is current)
        # (safe to skip; we always check before sampling)
        return

    def _rebuild_heap(self):
        """Rebuild heap from current ids using updated losses."""
        if not self.top_ids:
            self.top_heap = []
            self._heap_dirty = False
            return
        items = []
        for eid in list(self.top_ids):
            ep = self.episodes.get(eid)
            if ep is None:
                # stale id
                self.top_ids.discard(eid)
                continue
            items.append((float(ep.loss), int(eid)))
        # keep only top N by loss
        if len(items) > self.top_capacity:
            # nlargest then heapify the kept as a min-heap
            top_items = heapq.nlargest(self.top_capacity, items, key=lambda x: x[0])
            self.top_heap = top_items[:]  # will heapify next line
        else:
            self.top_heap = items[:]
        heapq.heapify(self.top_heap)
        # refresh set to match what we actually keep
        self.top_ids = {eid for _, eid in self.top_heap}
        self._heap_dirty = False

    def _episodes_to_batches(self, eps: List[Episode], batch_size: int):
        """Stream time-step minibatches [B,1,...] for a given list of episodes (process in chunks of batch_size)."""
        if not eps:
            return
        # process in windows of at most batch_size episodes
        for start in range(0, len(eps), max(1, batch_size)):
            group = eps[start:start + batch_size]
            for ep in group:
                ep.reset_cursor()
            ep_ids = [ep.id for ep in group]
            # time-step streaming
            while any(ep.load_t < ep.length for ep in group):
                O, A, R, NO, D = [], [], [], [], []
                for ep in group:
                    try:
                        o, a, r, no, d = ep.load()     # o: [1, obs_dim]; a: [1, act_dim]; r: [1]; d: [1]
                        O.append(o)                    # list of [1, obs_dim]
                        A.append(a)                    # list of [1, act_dim]
                        R.append(r)                    # list of [1]
                        NO.append(no)                  # list of [1, obs_dim]
                        D.append(d)                    # list of [1]
                    except StopIteration:
                        pass

                if not O:
                    break

                # Concatenate over batch -> final shapes
                obs_b      = th.cat(O,  dim=0)                # [B, obs_dim]
                act_b      = th.cat(A,  dim=0)                # [B, act_dim]
                next_obs_b = th.cat(NO, dim=0)                # [B, obs_dim]

                # r/d were 1D of length 1 per-ep; cat -> [B], then make column vectors -> [B, 1]
                rew_b      = th.cat(R, dim=0).unsqueeze(-1)   # [B, 1]
                done_b     = th.cat(D, dim=0).unsqueeze(-1)   # [B, 1]

                info = {"ep_ids": ep_ids}
                yield obs_b, act_b, rew_b, next_obs_b, done_b, info


    # ---------------- Add / Extend ----------------
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
        """Add one full trace as an episode. Always goes to FIFO; overflowed episode is inserted into the heap."""
        T = len(states)
        assert len(actions) == T and len(rewards) == T, "Episode arrays length mismatch."

        agg = _reward_agg_fn(reward_agg)
        rew_np = np.asarray([agg(_as_1d_float(r)) for r in rewards], dtype=np.float32)
        obs_np = np.stack([_as_1d_float(s) for s in states], axis=0).astype(np.float32)
        act_np = np.stack([_as_1d_float(a) for a in actions], axis=0).astype(np.float32)

        if T > 1:
            next_obs_np = np.vstack([obs_np[1:], np.zeros((1, obs_np.shape[1]), dtype=np.float32)])
        else:
            next_obs_np = np.zeros((1, obs_np.shape[1]), dtype=np.float32)

        done_np = np.array([float(network_output[t].get("done", 0)) for t in range(T)], dtype=np.float32)
        if enforce_last_done:
            done_np[-1] = 1.0

        eid = self._alloc_eid()
        ep = Episode(
            eid,
            obs_np, act_np, rew_np, next_obs_np, done_np,
            device=self.device,
            init_loss=100.0 if init_loss is None else float(init_loss),
        )
        self.episodes[eid] = ep

        # push into recent K; if overflow, promote the popped one into top-N heap
        overflow = None
        if len(self.recent_k) == self.recent_capacity:
            overflow = self.recent_k[0]  # record before it gets dropped
        self.recent_k.append(ep)
        if overflow is not None and overflow.id != ep.id:
            self._promote_to_heap(overflow)
        return eid

    @staticmethod
    def build_from_traces(
        traces,
        device: str = "cuda",
        reward_agg: str = "sum",
        n_envs: int = 1,
        recent_capacity: int = 10,
        top_capacity: int = 300,
        init_loss: Optional[float] = None,
        **kwargs,
    ):
        assert len(traces) >= 1, "No traces provided."
        first_states, first_actions, _, _ = traces[0]
        _ = _as_1d_float(first_states[0]).size
        _ = _as_1d_float(first_actions[0]).size

        buf = RNNPriReplayBuffer2.create(
            obs_dim=0, act_dim=0,  # dims unused here; kept for API parity
            device=device,
            recent_capacity=recent_capacity,
            top_capacity=top_capacity,
        )
        for (states, actions, rewards, network_output) in traces:
            buf.add_episode(
                states, actions, rewards, network_output,
                enforce_last_done=True, reward_agg=reward_agg, init_loss=init_loss
            )
        return buf

    def extend(
        self,
        traces,
        device: str = "cuda",
        reward_agg: str = "sum",
        init_loss: Optional[float] = None,
        **kwargs,
    ):
        """Process traces into the FIFO first; anything popped from the FIFO is inserted into the heap."""
        for (states, actions, rewards, network_output) in traces:
            self.add_episode(
                states, actions, rewards, network_output,
                enforce_last_done=True, reward_agg=reward_agg, init_loss=init_loss
            )

    # ---------------- Priority Updates ----------------
    def update_episode_losses(self, ep_ids: Iterable[int], losses: Iterable[float] | float):
        """Update losses; mark heap dirty so top-N is consistent before next sampling."""
        if isinstance(losses, float):
            id2loss = {int(i): losses for i in ep_ids}
        else:
            id2loss = {int(i): float(l) for i, l in zip(ep_ids, losses)}
        for eid, new_loss in id2loss.items():
            ep = self.episodes.get(int(eid))
            if ep is not None:
                ep.loss = float(new_loss)
                # If the episode is already in heap, we must rebuild the heap view.
                if eid in self.top_ids:
                    self._heap_dirty = True

    # ---------------- Minibatching ----------------
    def get_minibatches(self, batch_size: int, shuffle: bool = True):
        """
        同一批里优先使用当前 FIFO 中的全部可用 episode，再用 heap 补足到 batch_size。
        - b_fifo = min(当前FIFO剩余, batch_size)
        - b_heap = min(batch_size - b_fifo, 当前heap剩余)
        - 两侧都耗尽则结束；不做放回抽样，避免重复
        - 注意：随时间步推进，个别 episode 提前结束会导致 B 逐步变小（与 step-wise 训练一致）
        """
        if batch_size <= 0:
            return

        # --- 准备候选 ---
        recent_eps = list(self.recent_k)
        if shuffle:
            random.shuffle(recent_eps)

        if self._heap_dirty:
            self._rebuild_heap()
        heap_eids = list(self.top_ids)
        if shuffle:
            random.shuffle(heap_eids)
        heap_eps_all = [self.episodes[eid] for eid in heap_eids if eid in self.episodes]

        i_fifo, i_heap = 0, 0
        while True:
            # 关键两行：按你提的规则分配配额
            b_fifo = min(len(recent_eps) - i_fifo, batch_size)
            b_heap = min(batch_size - b_fifo, len(heap_eps_all) - i_heap)

            if b_fifo <= 0 and b_heap <= 0:
                break

            group = []
            if b_fifo > 0:
                group.extend(recent_eps[i_fifo:i_fifo + b_fifo])
                i_fifo += b_fifo
            if b_heap > 0:
                group.extend(heap_eps_all[i_heap:i_heap + b_heap])
                i_heap += b_heap

            # 重置游标并按时间步流式输出（不加时间轴）
            for ep in group:
                ep.reset_cursor()
            ep_ids = [ep.id for ep in group]

            
            while True:
                O, A, R, NO, D = [], [], [], [], []
                try:
                    for ep in group:
                        o, a, r, no, d = ep.load()  # o:[1,obs], a:[1,act], r:[1], d:[1]
                        O.append(o); A.append(a); R.append(r); NO.append(no); D.append(d)
                except Exception as e:
                    break
                    
                obs_b      = th.cat(O,  dim=0)               # [B, obs_dim]
                act_b      = th.cat(A,  dim=0)               # [B, act_dim]
                next_obs_b = th.cat(NO, dim=0)               # [B, obs_dim]
                rew_b      = th.cat(R,  dim=0).unsqueeze(-1) # [B, 1]
                done_b     = th.cat(D,  dim=0).unsqueeze(-1) # [B, 1]
                
                yield obs_b, act_b, rew_b, next_obs_b, done_b, {"ep_ids": ep_ids}