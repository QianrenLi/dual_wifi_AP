from dataclasses import dataclass
from typing import Optional
import torch as th

from net_util.advantage_estimator import ADV_REGISTRY, AdvEstimator

# -----------------------------
# Rollout buffer with GAE(λ)
# -----------------------------
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