from dataclasses import dataclass
from net_util import register_buffer
import torch as th
import numpy as np
from net_util.advantage_estimator import ADV_REGISTRY, AdvEstimator

# -----------------------------
# Helper
# -----------------------------
def _as_1d_float(x):
    """Coerce to 1-D float np.array without re-flattening dicts."""
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

# -----------------------------
# Rollout buffer with GAE(λ)
# -----------------------------
@register_buffer
@dataclass
class RolloutBuffer:
    obs: th.Tensor
    actions: th.Tensor
    log_probs: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    values: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    ptr: int = 0
    size: int = 0
    n_envs: int = 1
    advantage_estimator: str = "gae"  # default method

    @staticmethod
    def create(buffer_size: int, obs_dim: int, act_dim: int, n_envs: int, device):
        shape = (buffer_size, n_envs)
        obs = th.zeros(buffer_size, n_envs, obs_dim, device=device)
        actions = th.zeros(buffer_size, n_envs, act_dim, device=device)
        log_probs = th.zeros(buffer_size, n_envs, device=device)
        rewards = th.zeros(buffer_size, n_envs, device=device)
        dones = th.zeros(buffer_size, n_envs, device=device)
        values = th.zeros(buffer_size, n_envs, device=device)
        advantages = th.zeros(buffer_size, n_envs, device=device)
        returns = th.zeros(buffer_size, n_envs, device=device)
        return RolloutBuffer(obs, actions, log_probs, rewards, dones, values, advantages, returns, 0, buffer_size, n_envs)
    
    
    @staticmethod
    def build_from_traces(
        traces,                                # list of (states, actions, rewards, network_output)
        device="cuda",
        advantage_estimator="gae",
        gamma: float = 0.99,
        lam: float = 0.95,
        reward_agg="sum",  # "sum" | "mean" | callable(np.ndarray)->float
    ):
        """
        Concatenate multiple flat traces into ONE RolloutBuffer.
        Each element of `traces` is (states, actions, rewards, network_output),
        where states[t], actions[t], rewards[t] are already 1-D float lists (or array-likes),
        and network_output[t] may contain 'log_prob', 'value', optional 'done' and 'next_value'.
        We force an episode boundary at the end of each trace if 'done' is missing there.
        """
        import numpy as np
        # Reward aggregator
        if reward_agg == "sum":
            agg = lambda arr: float(np.asarray(arr, dtype=np.float32).sum())
        elif reward_agg == "mean":
            agg = lambda arr: float(np.asarray(arr, dtype=np.float32).mean())
        elif callable(reward_agg):
            agg = lambda arr: float(reward_agg(np.asarray(arr, dtype=np.float32)))
        else:
            raise ValueError("reward_agg must be 'sum', 'mean', or callable")

        # Basic dims from first timestep of first trace
        assert len(traces) >= 1, "No traces provided."
        first_states, first_actions, _, _ = traces[0]
        obs_dim = _as_1d_float(first_states[0]).size
        act_dim = _as_1d_float(first_actions[0]).size

        # Total steps
        total_T = sum(len(s) for (s, a, r, n) in traces)

        buf = RolloutBuffer.create(buffer_size=total_T, obs_dim=obs_dim, act_dim=act_dim, n_envs=1, device=device)
        buf.advantage_estimator = advantage_estimator

        # Fill buffer
        step_idx = 0
        last_value_for_bootstrap = 0.0  # fallback if nothing present
        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            assert len(actions) == T and len(rewards) == T and len(network_output) == T, "Trace lengths must match."

            for t in range(T):
                obs_t = th.tensor(_as_1d_float(states[t]),  dtype=th.float32, device=device).unsqueeze(0)
                # Prefer action from network_output (if you recorded it there), else use `actions`
                act_vec = _extract(network_output[t], ["action"], default=None)
                # if act_vec is None:
                #     act_vec = _as_1d_float(actions[t])
                act_t = th.tensor(act_vec, dtype=th.float32, device=device).unsqueeze(0)

                r_arr = _as_1d_float(rewards[t])
                rew_t = th.tensor([agg(r_arr)], dtype=th.float32, device=device)

                logp_vec = _extract(network_output[t], ["log_prob", "logp", "logprob", "log_probs"], default=None)
                logp_t  = th.tensor([_sum_logp(logp_vec)], dtype=th.float32, device=device)

                val_t   = _extract(network_output[t], ["value", "v"], default=0.0)
                val_t   = th.tensor([float(val_t)], dtype=th.float32, device=device)

                # Done handling: respect recorded 'done'; if missing, force done=True at the final step of this trace
                recorded_done = _extract(network_output[t], ["done", "terminal", "is_done"], default=None)
                if recorded_done is None:
                    done_flag = (t == T - 1)
                else:
                    done_flag = bool(recorded_done)
                    # if the trace doesn't mark last step as done, still keep what recorder says (you may override here if desired)
                    if t == T - 1 and recorded_done is None:
                        done_flag = True
                done_t  = th.tensor([1.0 if done_flag else 0.0], dtype=th.float32, device=device)

                buf.add(obs=obs_t, action=act_t, log_prob=logp_t, reward=rew_t, done=done_t, value=val_t)
                step_idx += 1

            # Track bootstrap candidate value for this trace (priority: explicit next/bootstrap → last value)
            last_val = _extract(network_output[-1], ["next_value", "bootstrap_value", "value", "v"], default=0.0)
            last_value_for_bootstrap = float(last_val)

        # Use the last seen bootstrap candidate
        last_val_t = th.tensor([last_value_for_bootstrap], dtype=th.float32, device=device)
        buf.compute_advantages(last_value=last_val_t, normalize=True, gamma=gamma, lam=lam)
        return buf
    

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_advantages(
        self,
        last_value: th.Tensor,                # [N]
        normalize: bool = True,
        **estimator_kwargs,                  # e.g. gamma=..., lam=...
    ) -> None:
        """
        Entry point to compute advantages/returns with pluggable estimators.
        - Choose a built-in by `method` OR pass a custom callable via `estimator`.
        - `normalize=True` applies per-update standardization to advantages.
        """
        if self.advantage_estimator not in ADV_REGISTRY:
            raise ValueError(f"Unknown advantage method '{self.advantage_estimator}'. "
                                f"Available: {list(ADV_REGISTRY.keys())} or pass a custom estimator.")
        estimator = ADV_REGISTRY[self.advantage_estimator]

        adv, ret = estimator(self.rewards, self.values, self.dones, last_value, **estimator_kwargs)

        if normalize:
            flat = adv.view(-1)
            adv = (adv - flat.mean()) / (flat.std() + 1e-8)

        self.advantages = adv
        self.returns = ret
        
        print(self.advantages.shape)

    def get_minibatches(self, batch_size: int):
        # Flatten T and N
        T, N = self.size, self.n_envs
        idx = th.randperm(T * N, device=self.obs.device)
        obs = self.obs.view(T * N, -1)
        actions = self.actions.view(T * N, -1)
        log_probs = self.log_probs.view(T * N)
        advantages = self.advantages.view(T * N)
        returns = self.returns.view(T * N)
        values = self.values.view(T * N)
        for start in range(0, T * N, batch_size):
            end = start + batch_size
            mb_idx = idx[start:end]
            yield obs[mb_idx], actions[mb_idx], log_probs[mb_idx], advantages[mb_idx], returns[mb_idx], values[mb_idx]

# -----------------------------