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


def _normalize_trace_paths(raw_paths):
    # raw_paths is a list (because nargs='+'); each item may itself be "a,b,c"
    paths = []
    for item in raw_paths:
        paths.extend([s for s in (p.strip() for p in item.split(",")) if s])
    # (Optional) de-dup while preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq



def build_rollout_buffer_from_traces_flat(
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
            if act_vec is None:
                act_vec = _as_1d_float(actions[t])
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


def main():
    p = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
    p.add_argument("--control_config", type=str, required=True)
    p.add_argument("--trace_path", type=str, required=True, nargs="+")
    p.add_argument("--load_path", type=str, default=None)
    args = p.parse_args()

    trace_paths = _normalize_trace_paths(args.trace_path)

    with open(args.control_config, 'r', encoding='utf-8') as f:
        control_config = json.load(f)

    policy_cfg = control_config['policy_cfg']
    policy_name = policy_cfg["policy_name"]
    policy_cfg_cls = POLICY_CFG_REGISTRY[policy_name]
    policy_cls = POLICY_REGISTRY[policy_name]

    rollout_cfg = control_config['rollout_cfg']

    # ---- load & flatten multiple traces
    merged_traces = []
    for tp in trace_paths:
        s, a, r, net = trace_collec(
            tp,
            state_descriptor=control_config.get('state_cfg', None),
            reward_descriptor=control_config.get('reward_cfg', None)
        )

        s = [flatten_leaves(x) for x in s]
        a = [flatten_leaves(x) for x in a]
        r = [flatten_leaves(x) for x in r]
        merged_traces.append((s, a, r, net))

    rollout_buffer = build_rollout_buffer_from_traces_flat(
        merged_traces,
        device="cuda",
        advantage_estimator=rollout_cfg["advantage_estimator"],
        gamma=rollout_cfg["gamma"],
        lam=rollout_cfg["lam"],
        reward_agg=rollout_cfg.get("reward_agg", "sum"),
    )

    policy_cfg = _inflate_dataclass_from_manifest(policy_cfg_cls, policy_cfg)
    policy: PolicyBase = policy_cls(ControlCmd, policy_cfg, rollout_buffer=rollout_buffer, device="cuda")

    if args.load_path is None or args.load_path.lower() == "none":
        store_path = Path("net_util/net_cp") / Path(args.control_config).parent.stem / "1.pt"
    else:
        policy.load(args.load_path, device='cuda')
        next_id = int(Path(args.load_path).stem) + 1
        store_path = Path(args.load_path).parent / f"{next_id}.pt"

    os.makedirs(store_path.parent, exist_ok=True)
    print(store_path)

    policy.train_per_epoch()
    policy.save(store_path)
    

if __name__ == "__main__":
    main()