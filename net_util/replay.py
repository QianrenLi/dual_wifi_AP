from dataclasses import dataclass
from net_util import register_buffer
import torch as th
import numpy as np

# -----------------------------
# Helper
# -----------------------------
def _as_1d_float(x):
    """Coerce to 1-D float np.array without re-flattening dicts."""
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1)

@register_buffer
@dataclass
class ReplayBuffer:
    obs: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    next_obs: th.Tensor
    dones: th.Tensor
    ptr: int = 0
    size: int = 0

    @staticmethod
    def create(buffer_size: int, obs_dim: int, act_dim: int, n_envs: int, device):
        obs = th.zeros(buffer_size, n_envs, obs_dim, device=device)
        actions = th.zeros(buffer_size, n_envs, act_dim, device=device)
        rewards = th.zeros(buffer_size, n_envs, device=device)
        next_obs = th.zeros(buffer_size, n_envs, obs_dim, device=device)
        dones = th.zeros(buffer_size, n_envs, device=device)
        return ReplayBuffer(obs, actions, rewards, next_obs, dones, 0, buffer_size)
    
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr += 1
        
    def get_minibatches(self, batch_size: int):
        # Flatten T and N
        T = self.size
        idx = th.randperm(T, device=self.obs.device)
        obs = self.obs.view(T, -1)
        actions = self.actions.view(T, -1)
        rewards = self.rewards.view(T, -1)
        next_obs = self.next_obs.view(T, -1)
        dones = self.dones.view(T, -1)
        for start in range(0, T , batch_size):
            end = start + batch_size
            mb_idx = idx[start:end]
            yield obs[mb_idx], actions[mb_idx], rewards[mb_idx], next_obs[mb_idx], dones[mb_idx]
            
    @staticmethod
    def build_from_traces(
        traces,  # list of (states, actions, rewards, network_output)
        device="cuda",
        reward_agg="sum",  # "sum" | "mean" | callable(np.ndarray)->float
        **kwargs,
    ):
        """
        Concatenate multiple flat traces into ONE ReplayBuffer.
        Each element of `traces` is (states, actions, rewards, network_output),
        where states[t], actions[t], rewards[t] are already 1-D float lists (or array-likes),
        and network_output[t] may contain 'done' flags.
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

        buf = ReplayBuffer.create(buffer_size=total_T, obs_dim=obs_dim, act_dim=act_dim, n_envs=1, device=device)

        # Fill buffer
        step_idx = 0
        for (states, actions, rewards, network_output) in traces:
            T = len(states)
            assert len(actions) == T and len(rewards) == T and len(network_output) == T, "Trace lengths must match."

            for t in range(T):
                # Convert each step into tensors and add to the buffer
                obs_t = th.tensor(_as_1d_float(states[t]), dtype=th.float32, device=device).unsqueeze(0)
                act_t = th.tensor(_as_1d_float(actions[t]), dtype=th.float32, device=device).unsqueeze(0)
                rew_t = th.tensor([agg(_as_1d_float(rewards[t]))], dtype=th.float32, device=device)
                
                # For next_obs_t, get the state at the next timestep (obs_{t+1})
                if t + 1 < T:
                    next_obs_t = th.tensor(_as_1d_float(states[t + 1]), dtype=th.float32, device=device).unsqueeze(0)
                else:
                    # For the last timestep, the next observation is zero (or another convention you prefer)
                    next_obs_t = th.zeros(1, obs_dim, device=device)
                    
                done_t = th.tensor([float(network_output[t].get("done", 0))], dtype=th.float32, device=device)

                buf.add(obs=obs_t, action=act_t, reward=rew_t, next_obs=next_obs_t, done=done_t)
                step_idx += 1
        
        return buf
