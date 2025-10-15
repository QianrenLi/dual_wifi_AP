import json
import os
import argparse
from pathlib import Path
from net_util import POLICY_REGISTRY, POLICY_CFG_REGISTRY, BUFFER_REGISTRY
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
    Buffer = BUFFER_REGISTRY[rollout_cfg['buffer_name']]

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

    rollout_buffer = Buffer.build_from_traces(
        merged_traces,
        device="cuda",
        advantage_estimator=rollout_cfg.get("advantage_estimator", None),
        gamma=rollout_cfg.get("gamma", None),
        lam=rollout_cfg.get("lam", None),
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
    import importlib.util
    config_path = Path('/home/qianren/workspace/dual_wifi_AP/config/rtt_only.py')
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    policy_configs =cfg.policy_config()
    
    network_folder = f"net_util/net_config/test"
    os.makedirs(network_folder, exist_ok=True)
    
    for src, config in policy_configs.items():
        for des, net_config in config.items():
            data_path = f"{network_folder}/test.json"
            with open(data_path, "w") as f:
                f.write(json.dumps(net_config, indent=4))
                
    main()