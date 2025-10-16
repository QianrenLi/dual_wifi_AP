# train_rl.py
import argparse, json, os, time
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

import torch as th
from torch.utils.tensorboard import SummaryWriter

from net_util import POLICY_REGISTRY, POLICY_CFG_REGISTRY, BUFFER_REGISTRY
from net_util.base import PolicyBase
from util.control_cmd import ControlCmd
from util.trace_watcher import TraceWatcher


# ----------------------------- Helpers -----------------------------
def _coerce_to_type(val: Any, typ) -> Any:
    o, a = get_origin(typ), get_args(typ)
    if o is Union and type(None) in a:
        inner = next(t for t in a if t is not type(None))
        return None if val is None else _coerce_to_type(val, inner)
    if o is list:
        (inner,) = a or (Any,)
        return [] if val is None else [_coerce_to_type(x, inner) for x in val]
    if o is dict:
        k_t, v_t = a or (Any, Any)
        return {} if val is None else { _coerce_to_type(k, k_t): _coerce_to_type(v, v_t) for k, v in val.items() }
    if typ is Path: return val if isinstance(val, Path) else Path(val)
    if typ is bool: return val.strip().lower() in ("1","true","yes","y","on") if isinstance(val,str) else bool(val)
    if typ is int:  return int(val)
    if typ is float:return float(val)
    return val

def _inflate_dataclass_from_manifest(cls, manifest: Dict[str, Any]) -> Any:
    vals = {}
    for f in fields(cls):
        raw = manifest.get(f.name, f.default if f.default is not MISSING
                           else f.default_factory() if getattr(f,"default_factory",MISSING) is not MISSING else None)
        if raw is None and f.name not in manifest:
            raise ValueError(f"Missing required field '{f.name}' for {cls.__name__}")
        vals[f.name] = _coerce_to_type(raw, f.type)
    return cls(**vals)

def _flatten_params(m: Optional[th.nn.Module]) -> th.Tensor:
    if m is None: return th.zeros(0)
    parts = [p.detach().view(-1).cpu() for p in m.parameters() if p.requires_grad]
    return th.cat(parts) if parts else th.zeros(0)


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
    ap.add_argument("--control_config", required=True)
    ap.add_argument("--trace_path", required=True, help="Root folder to watch; all **/*.jsonl included")
    ap.add_argument("--load_path", default=None)
    ap.add_argument("--delta-min", type=float, default=5e-4)
    ap.add_argument("--delta-max", type=float, default=10.0)
    args = ap.parse_args()

    with open(args.control_config, "r", encoding="utf-8") as f:
        control_cfg = json.load(f)

    # Registries
    pol_manifest = control_cfg["policy_cfg"]
    pol_name = pol_manifest["policy_name"]
    PolicyCfg = POLICY_CFG_REGISTRY[pol_name]
    PolicyCls = POLICY_REGISTRY[pol_name]
    roll_cfg = control_cfg["rollout_cfg"]
    BufferCls = BUFFER_REGISTRY[roll_cfg["buffer_name"]]

    # Trace watcher (single root, recursive *.jsonl)
    watcher = TraceWatcher(args.trace_path, control_cfg)
    init_traces = watcher.load_initial_traces()
    while init_traces == []:
        init_traces = watcher.load_initial_traces()
        time.sleep(1)
    
    # Build buffer
    buf = BufferCls.build_from_traces(
        init_traces,
        device="cuda",
        advantage_estimator=roll_cfg.get("advantage_estimator"),
        gamma=roll_cfg.get("gamma"),
        lam=roll_cfg.get("lam"),
        reward_agg=roll_cfg.get("reward_agg", "sum"),
        buffer_max=roll_cfg.get("buffer_max", 100_000),
    )

    # Policy
    pol_cfg = _inflate_dataclass_from_manifest(PolicyCfg, pol_manifest)
    policy: PolicyBase = PolicyCls(ControlCmd, pol_cfg, rollout_buffer=buf, device="cuda")

    # Checkpoint pathing
    cfg_stem = Path(args.control_config).parent.stem
    if not args.load_path or str(args.load_path).lower() == "none":
        ckpt_dir = Path("net_util/net_cp") / cfg_stem
        next_id = 1
    else:
        policy.load(args.load_path, device="cuda")
        ckpt_dir = Path(args.load_path).parent
        next_id = int(Path(args.load_path).stem) + 1
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    store_path = ckpt_dir / f"{next_id}.pt"
    latest_path = ckpt_dir / f"latest.pt"

    # TB + training
    writer = SummaryWriter(f"net_util/logs/{cfg_stem}")
    actor_before = _flatten_params(getattr(policy, "actor", None))

    def _extend_with_new():
        new_traces = watcher.poll_new_traces()
        if new_traces:
            policy.buf.extend(
                new_traces,
                device="cuda",
                reward_agg=roll_cfg.get("reward_agg", "sum"),
                advantage_estimator=roll_cfg.get("advantage_estimator"),
                gamma=roll_cfg.get("gamma"),
                lam=roll_cfg.get("lam"),
            )
            writer.add_scalar("traces/new_count", len(new_traces), getattr(policy, "_global_step", 0))
            return True
        return False

    epoch = 0
    while True:
        policy.train_per_epoch(epoch, writer=writer)
        # Extend when new data appears
        _extend_with_new()

        # Actor drift
        actor_after = _flatten_params(getattr(policy, "actor", None))
        # n = min(actor_before.numel(), actor_after.numel())
        delta = float((actor_after - actor_before).norm(p=2).item())
        writer.add_scalar("params/delta_actor_epoch_L2", delta, epoch)
        # If changes are tiny, block until a new trace arrives, then extend once
        if delta <= args.delta_min:
            while not _extend_with_new():
                time.sleep(1)

        # Big jump → save and roll checkpoint id
        if delta >= args.delta_max:
            actor_before = actor_after
            policy.save(store_path)
            policy.save(latest_path)
            next_id += 1
            store_path = ckpt_dir / f"{next_id}.pt"

        epoch += 1


if __name__ == "__main__":
    # Example: generate test config files then run
    import importlib.util
    cfg_path = Path("/home/qianren/workspace/dual_wifi_AP/config/rtt_only.py")
    spec = importlib.util.spec_from_file_location("exp_config", cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg_mod)
    policy_configs = cfg_mod.policy_config()

    net_dir = Path("net_util/net_config/test"); net_dir.mkdir(parents=True, exist_ok=True)
    with open(net_dir / "test.json", "w") as f:
        json.dump(next(iter(next(iter(policy_configs.values())).values())), f, indent=4)

    main()
