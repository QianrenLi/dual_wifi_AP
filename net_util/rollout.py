from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from net_util import register_buffer
from net_util.advantage_estimator import ADV_REGISTRY

# -----------------------------
# Helpers
# -----------------------------
def _as_1d_float(x):
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)

def _extract(netout_step, keys, default=None):
    if not isinstance(netout_step, dict):
        return default
    for k in keys:
        if k in netout_step:
            return netout_step[k]
    return default

def _sum_logp(x):
    if x is None:
        return 0.0
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    return float(np.asarray(x, dtype=np.float32).sum())

def _reward_agg_fn(reward_agg):
    if reward_agg == "sum":  return lambda arr: float(np.asarray(arr, dtype=np.float32).sum())
    if reward_agg == "mean": return lambda arr: float(np.asarray(arr, dtype=np.float32).mean())
    if callable(reward_agg): return lambda arr: float(reward_agg(np.asarray(arr, dtype=np.float32)))
    raise ValueError("reward_agg must be 'sum', 'mean', or callable")

# -----------------------------
# Rollout buffer (ring) with GAE(λ) + TensorBoard hooks
# -----------------------------
@register_buffer
@dataclass
class RolloutBuffer:
    # Storage shapes:
    #   obs, next_obs: [capacity, n_envs, obs_dim]
    #   actions:       [capacity, n_envs, act_dim]
    #   log_probs:     [capacity, n_envs]
    #   rewards:       [capacity, n_envs]
    #   dones:         [capacity, n_envs]
    #   values:        [capacity, n_envs]
    #   advantages:    [capacity, n_envs]
    #   returns:       [capacity, n_envs]
    obs: th.Tensor
    actions: th.Tensor
    log_probs: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    values: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    ptr: int = 0         # next physical write index in ring
    size: int = 0        # number of valid rows (<= capacity)
    n_envs: int = 1
    advantage_estimator: str = "gae"

    # Optional TensorBoard
    _writer: Optional[SummaryWriter] = None
    _tb_step: int = 0
    _tb_every: int = 100  # log buffer stats every N .add() calls

    # ---------- Construction ----------
    @staticmethod
    def create(buffer_size: int, obs_dim: int, act_dim: int, n_envs: int, device,
               *, writer: Optional[SummaryWriter] = None, tb_every: int = 100):
        obs         = th.zeros(buffer_size, n_envs, obs_dim, device=device)
        actions     = th.zeros(buffer_size, n_envs, act_dim, device=device)
        log_probs   = th.zeros(buffer_size, n_envs, device=device)
        rewards     = th.zeros(buffer_size, n_envs, device=device)
        dones       = th.zeros(buffer_size, n_envs, device=device)
        values      = th.zeros(buffer_size, n_envs, device=device)
        advantages  = th.zeros(buffer_size, n_envs, device=device)
        returns     = th.zeros(buffer_size, n_envs, device=device)
        # start empty: size=0, ptr=0  (capacity is allocated memory)
        return RolloutBuffer(
            obs, actions, log_probs, rewards, dones, values, advantages, returns,
            0, 0, n_envs, "gae", writer, 0, tb_every
        )

    # ---------- Convenience ----------
    @property
    def capacity(self) -> int:
        return self.obs.shape[0]

    def set_writer(self, writer: Optional[SummaryWriter], tb_every: Optional[int] = None):
        self._writer = writer
        if tb_every is not None:
            self._tb_every = int(tb_every)

    # ---------- Ring index mapping ----------
    def _logical_to_physical(self, logical_idx: th.Tensor) -> th.Tensor:
        # logical indices are in [0, size); 0 is the oldest element
        if self.size == 0:
            return th.empty_like(logical_idx)
        start = (self.ptr - self.size) % self.capacity
        return (start + logical_idx) % self.capacity

    # ---------- Insertion (ring write) ----------
    def add(self, obs, action, log_prob, reward, done, value):
        """
        Insert one step for all envs (expects shapes [n_envs, ...]).
        Overwrites oldest rows when full.
        """
        idx = self.ptr
        self.obs[idx]        = obs
        self.actions[idx]    = action
        self.log_probs[idx]  = log_prob
        self.rewards[idx]    = reward
        self.dones[idx]      = done
        self.values[idx]     = value

        # advance ring pointer, grow size up to capacity
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

        # light TB logging
        if self._writer is not None:
            self._tb_step += 1
            if (self._tb_step % self._tb_every) == 0:
                self._writer.add_scalar("buffer/ptr", self.ptr, self._tb_step)
                self._writer.add_scalar("buffer/size", self.size, self._tb_step)
                self._writer.add_scalar("buffer/capacity", self.capacity, self._tb_step)

    # ---------- Build from traces (capacity-bounded, cyclic) ----------
    @staticmethod
    def build_from_traces(
        traces,                                # list of (states, actions, rewards, network_output)
        device="cuda",
        advantage_estimator="gae",
        gamma: float = 0.99,
        lam: float = 0.95,
        reward_agg="sum",
        buffer_max: int = 100_000,
        n_envs: int = 1,
        writer: Optional[SummaryWriter] = None,
        tb_every: int = 100,
    ):
        assert len(traces) >= 1, "No traces provided."
        agg = _reward_agg_fn(reward_agg)

        first_states, first_actions, _, _ = traces[0]
        obs_dim = _as_1d_float(first_states[0]).size
        act_dim = _as_1d_float(first_actions[0]).size

        buf = RolloutBuffer.create(
            buffer_size=int(buffer_max), obs_dim=obs_dim, act_dim=act_dim,
            n_envs=n_envs, device=device, writer=writer, tb_every=tb_every
        )
        buf.advantage_estimator = advantage_estimator

        # stream inserts
        last_value_for_bootstrap = 0.0
        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            assert len(actions) == T and len(rewards) == T and len(network_output) == T, "Trace lengths must match."

            for t in range(T):
                # obs / action (prefer recorded action in netout)
                obs_t = th.tensor(_as_1d_float(states[t]),  dtype=th.float32, device=device).unsqueeze(0)
                act_vec = _extract(network_output[t], ["action"], default=_as_1d_float(actions[t]))
                act_t = th.tensor(act_vec, dtype=th.float32, device=device).unsqueeze(0)

                # reward (aggregated per step)
                r_arr = _as_1d_float(rewards[t])
                rew_t = th.tensor([agg(r_arr)], dtype=th.float32, device=device)

                # log prob
                logp_vec = _extract(network_output[t], ["log_prob", "logp", "logprob", "log_probs"], default=None)
                logp_t  = th.tensor([_sum_logp(logp_vec)], dtype=th.float32, device=device)

                # value
                val_t   = _extract(network_output[t], ["value", "v"], default=0.0)
                val_t   = th.tensor([float(val_t)], dtype=th.float32, device=device)

                # done
                recorded_done = _extract(network_output[t], ["done", "terminal", "is_done"], default=None)
                done_flag = bool(recorded_done) if recorded_done is not None else (t == T - 1)
                done_t  = th.tensor([1.0 if done_flag else 0.0], dtype=th.float32, device=device)

                buf.add(obs=obs_t, action=act_t, log_prob=logp_t, reward=rew_t, done=done_t, value=val_t)

            # bootstrap preference for this trace
            last_val = _extract(network_output[-1], ["next_value", "bootstrap_value", "value", "v"], default=0.0)
            last_value_for_bootstrap = float(last_val)

        # compute advantages over current logical window
        last_val_t = th.tensor([last_value_for_bootstrap], dtype=th.float32, device=device)
        buf.compute_advantages(last_value=last_val_t, normalize=True, gamma=gamma, lam=lam)
        return buf

    # ---------- Advantages over logical window (handles wrap) ----------
    def compute_advantages(
        self,
        last_value: th.Tensor,                # [N]
        normalize: bool = True,
        **estimator_kwargs,                  # gamma=..., lam=..., etc.
    ) -> None:
        if self.advantage_estimator not in ADV_REGISTRY:
            raise ValueError(f"Unknown advantage method '{self.advantage_estimator}'. "
                             f"Available: {list(ADV_REGISTRY.keys())}")
        estimator = ADV_REGISTRY[self.advantage_estimator]

        T, N = self.size, self.n_envs
        if T == 0:
            return

        # Build logical index [0..T-1] and map to physical rows (handles wrap)
        logical_idx = th.arange(T, device=self.obs.device)
        phys_idx = self._logical_to_physical(logical_idx)

        # Gather rewards/values/dones in logical order [T, N]
        rew   = self.rewards.index_select(0, phys_idx)
        val   = self.values.index_select(0, phys_idx)
        dones = self.dones.index_select(0, phys_idx)

        # Compute advantages/returns on the logical sequence
        adv, ret = estimator(rew, val, dones, last_value, **estimator_kwargs)

        # Normalize (per-update) if requested
        if normalize:
            flat = adv.view(-1)
            adv = (adv - flat.mean()) / (flat.std() + 1e-8)

        # Scatter back to physical rows
        self.advantages.index_copy_(0, phys_idx, adv)
        self.returns.index_copy_(0, phys_idx, ret)

        # TensorBoard stats
        if self._writer is not None:
            self._tb_step += 1
            self._writer.add_scalar("buffer/size", self.size, self._tb_step)
            self._writer.add_scalar("buffer/ptr", self.ptr, self._tb_step)
            self._writer.add_scalar("advantages/mean", adv.mean().item(), self._tb_step)
            self._writer.add_scalar("advantages/std", adv.std(unbiased=False).item(), self._tb_step)
            self._writer.add_scalar("returns/mean", ret.mean().item(), self._tb_step)

    # ---------- Minibatch sampler (uniform over logical window) ----------
    def get_minibatches(self, batch_size: int):
        """
        Uniformly shuffle and yield minibatches from the VALID logical window [0..size).
        Works regardless of ring wrap position.
        """
        T, N = self.size, self.n_envs
        if T == 0:
            return

        # logical row indices and mapping to physical
        logical_rows = th.arange(T, device=self.obs.device)
        phys_rows = self._logical_to_physical(logical_rows)

        # materialize the logical view (contiguous [T, ...]) then flatten TN for indexing
        obs      = self.obs.index_select(0, phys_rows).reshape(T * N, -1)
        actions  = self.actions.index_select(0, phys_rows).reshape(T * N, -1)
        logp     = self.log_probs.index_select(0, phys_rows).reshape(T * N)
        adv      = self.advantages.index_select(0, phys_rows).reshape(T * N)
        ret      = self.returns.index_select(0, phys_rows).reshape(T * N)
        vals     = self.values.index_select(0, phys_rows).reshape(T * N)

        perm = th.randperm(T * N, device=self.obs.device)
        for start in range(0, T * N, batch_size):
            end = min(start + batch_size, T * N)
            mb_idx = perm[start:end]
            yield obs[mb_idx], actions[mb_idx], logp[mb_idx], adv[mb_idx], ret[mb_idx], vals[mb_idx]
