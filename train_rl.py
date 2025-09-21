import json
import os
import argparse
from pathlib import Path
from net_util import POLICY_REGISTRY, POLICY_CFG_REGISTRY
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, Union
from dataclasses import dataclass, MISSING, fields, is_dataclass
from net_util.base import PolicyBase
from util.control_cmd import ControlCmd, cmd_to_list
from util.trace_collec import trace_collec, flatten_leaves
from net_util.rollout import RolloutBuffer

def _coerce_to_type(val: Any, typ) -> Any:
    origin = get_origin(typ)
    args = get_args(typ)

    # Optional[T] -> Union[T, NoneType]
    if origin is Union and type(None) in args:
        inner = next(t for t in args if t is not type(None))
        return None if val is None else _coerce_to_type(val, inner)

    # Containers
    if origin is list:
        (inner,) = args or (Any,)
        return [] if val is None else [ _coerce_to_type(x, inner) for x in val ]
    if origin is dict:
        (k_t, v_t) = args or (Any, Any)
        return {} if val is None else { _coerce_to_type(k, k_t): _coerce_to_type(v, v_t) for k, v in val.items() }

    # Concrete types
    if typ is Path:
        return Path(val) if not isinstance(val, Path) else val
    if typ is bool:
        # handle "true"/"false"/1/0 robustly
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(val)
    if typ is int:
        return int(val)
    if typ is float:
        return float(val)
    # str or Any or already correct
    return val

def _inflate_dataclass_from_manifest(cls, manifest: Dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict using annotations + defaults."""
    values: Dict[str, Any] = {}
    for f in fields(cls):
        # pick value: manifest -> dataclass default -> default_factory -> global defaults
        if f.name in manifest:
            raw = manifest[f.name]
        elif f.default is not MISSING:
            raw = f.default
        elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
            raw = f.default_factory()           # type: ignore[misc]
        else:
            raise ValueError(f"Missing required field '{f.name}' for {cls.__name__}")
        
        values[f.name] = _coerce_to_type(raw, f.type)
    return cls(**values)


import numpy as np
import torch as th

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


def build_rollout_buffer_from_trace_flat(
    states, actions, rewards, network_output,
    device="cuda", advantage_estimator="gae",
    gamma: float = 0.99, lam: float = 0.95,
    reward_agg="sum",  # "sum" | "mean" | callable(np.ndarray)->float
):
    """
    states[t], actions[t], rewards[t] are already 1-D float lists (or array-likes).
    network_output[t] is a dict, may contain:
      - 'log_prob' (scalar or per-dim vector)
      - 'value' (scalar)
      - optional 'done'
      - optional 'next_value'/'bootstrap_value' for bootstrapping
    """
    T = len(states)
    assert len(actions) == T and len(rewards) == T and len(network_output) == T, "Trace lengths must match."

    obs_dim = _as_1d_float(states[0]).size
    act_dim = _as_1d_float(actions[0]).size

    buf = RolloutBuffer.create(buffer_size=T, obs_dim=obs_dim, act_dim=act_dim, n_envs=1, device=device)
    buf.advantage_estimator = advantage_estimator

    # Reward aggregator
    if reward_agg == "sum":
        agg = lambda arr: float(arr.sum())
    elif reward_agg == "mean":
        agg = lambda arr: float(arr.mean())
    elif callable(reward_agg):
        agg = lambda arr: float(reward_agg(arr))
    else:
        raise ValueError("reward_agg must be 'sum', 'mean', or callable")

    for t in range(T):
        obs_t = th.tensor(_as_1d_float(states[t]),  dtype=th.float32, device=device).unsqueeze(0)   # [1, obs_dim]
        # act_t = th.tensor(_as_1d_float(actions[t]), dtype=th.float32, device=device).unsqueeze(0)   # [1, act_dim]

        # rewards already 1-D list → aggregate to scalar
        r_arr = _as_1d_float(rewards[t])
        rew_t = th.tensor([agg(r_arr)], dtype=th.float32, device=device)                            # [1]

        net_t = network_output[t]
        
        act_vec = _extract(net_t, ["action"], default=None)
        act_t = th.tensor(act_vec, dtype=th.float32, device=device).unsqueeze(0)
        
        logp_vec = _extract(net_t, ["log_prob", "logp", "logprob", "log_probs"], default=None)
        logp_t  = th.tensor([_sum_logp(logp_vec)], dtype=th.float32, device=device)                 # [1]

        val_t   = _extract(net_t, ["value", "v"], default=0.0)
        val_t   = th.tensor([float(val_t)], dtype=th.float32, device=device)                        # [1]

        done_t  = _extract(net_t, ["done", "terminal", "is_done"], default=False)
        done_t  = th.tensor([1.0 if bool(done_t) else 0.0], dtype=th.float32, device=device)        # [1]

        buf.add(obs=obs_t, action=act_t, log_prob=logp_t, reward=rew_t, done=done_t, value=val_t)

    # Bootstrap with next/bootstrapped value if present, else last value, else 0
    last_val = _extract(network_output[-1], ["next_value", "bootstrap_value", "value", "v"], default=0.0)
    last_val = th.tensor([float(last_val)], dtype=th.float32, device=device)

    buf.compute_advantages(last_value=last_val, normalize=True, gamma=gamma, lam=lam)
    return buf


def main():
    p = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
    p.add_argument("--control_config", type=str, required=True)
    p.add_argument("--trace_path", type=str, required=True)
    p.add_argument("--load_path", type=str, default= None)
    
    args = p.parse_args()

    with open(args.control_config, 'r', encoding='utf-8') as f:
        control_config = json.load(f)
    
    policy_cfg = control_config['policy_cfg']
    policy_name = policy_cfg["policy_name"]
    policy_cfg_cls = POLICY_CFG_REGISTRY[policy_name]
    policy_cls = POLICY_REGISTRY[policy_name]
    
    rollout_cfg = control_config['rollout_cfg']
    states, actions, rewards, network_output = trace_collec(args.trace_path, state_descriptor=control_config.get('state_cfg', None), reward_descriptor=control_config.get('reward_cfg', None))
    
    states = [ flatten_leaves(state) for state in states ]
    actions = [ flatten_leaves(action) for action in actions ]
    rewards = [ flatten_leaves(reward) for reward in rewards ]
    
    rollout_buffer = build_rollout_buffer_from_trace_flat( states, actions, rewards, network_output, advantage_estimator=rollout_cfg["advantage_estimator"], gamma=rollout_cfg["gamma"], lam=rollout_cfg["lam"])
    
    policy_cfg = _inflate_dataclass_from_manifest(policy_cfg_cls, policy_cfg)
    policy: PolicyBase = policy_cls( ControlCmd, policy_cfg, rollout_buffer=rollout_buffer, device = "cuda")
    
    if args.load_path is None or args.load_path == "none":
        store_path = Path("net_util/net_cp") / Path(args.control_config).parent.stem / "1.pt"
    else:
        policy.load(args.load_path, device = 'cuda')
        next_id = int(Path(args.load_path).stem) + 1
        store_path = Path(args.load_path).parent / f"{next_id}.pt"
        
    os.makedirs(store_path.parent, exist_ok=True)
    print(store_path)
    
    policy.train_per_epoch()
    policy.save(store_path)
    

if __name__ == "__main__":
    main()