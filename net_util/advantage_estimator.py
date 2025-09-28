from typing import Callable, Dict, Tuple
import torch as th

AdvEstimator = Callable[
    [th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    Tuple[th.Tensor, th.Tensor]
]
# signature: (rewards[T,N], values[T,N], dones[T,N], last_value[N]) -> (advantages[T,N], returns[T,N])

# ---------- Detached estimators ----------
def gae(
    rewards: th.Tensor, values: th.Tensor, dones: th.Tensor, last_value: th.Tensor,
    gamma: float = 0.99, lam: float = 0.95
) -> Tuple[th.Tensor, th.Tensor]:
    T, N = rewards.shape
    adv = th.zeros_like(rewards)
    last_gae = th.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret

def td0(
    rewards: th.Tensor, values: th.Tensor, dones: th.Tensor, last_value: th.Tensor,
    gamma: float = 0.99, **estimator_kwargs
) -> Tuple[th.Tensor, th.Tensor]:
    T, N = rewards.shape
    adv = th.zeros_like(rewards)
    ret = th.zeros_like(rewards)
    for t in range(T):
        nonterminal = 1.0 - dones[t]
        next_v = last_value if t == T - 1 else values[t + 1]
        td_target = rewards[t] + gamma * next_v * nonterminal
        adv[t] = td_target - values[t]
        ret[t] = td_target
    return adv, ret

def mc(
    rewards: th.Tensor, values: th.Tensor, dones: th.Tensor, last_value: th.Tensor,
    gamma: float = 0.99, **estimator_kwargs
) -> Tuple[th.Tensor, th.Tensor]:
    T, N = rewards.shape
    ret = th.zeros_like(rewards)
    g = last_value  # bootstrap after T-1
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        g = rewards[t] + gamma * g * nonterminal
        ret[t] = g
    adv = ret - values
    return adv, ret

# Optional: a simple registry
ADV_REGISTRY: Dict[str, AdvEstimator] = {
    "gae": gae,
    "td0": td0,
    "mc": mc,
}