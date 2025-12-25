# train_rl.py
import argparse, json, os, time
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin
from datetime import datetime
import traceback

import torch as th
from torch.utils.tensorboard import SummaryWriter

from net_util import POLICY_REGISTRY, POLICY_CFG_REGISTRY, BUFFER_REGISTRY
from net_util.base import PolicyBase
# Ensure all algorithms are imported
from util.control_cmd import ControlCmd
from util.trace_watcher import TraceWatcher
from typing import List


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
    if isinstance(m , list):
        parts = [p.detach().view(-1).cpu() for p in m if p.requires_grad]
        return th.cat(parts) if parts else th.zeros(0)
    parts = [p.detach().view(-1).cpu() for p in m.parameters() if p.requires_grad]
    return th.cat(parts) if parts else th.zeros(0)

def _normalize_trace_paths(arg) -> List[str]:
    """
    Accept either:
      - list via argparse nargs='+' (e.g., ["A", "B"])
      - entries that may themselves be comma-separated (e.g., "A,B")
      - a single string "A,B" if user passes it as one token
    Returns a flat list of non-empty strings.
    """
    if isinstance(arg, str):
        return [p for p in (s.strip() for s in arg.split(",")) if p]
    out: List[str] = []
    for item in arg:
        parts = [p.strip() for p in str(item).split(",")]
        out.extend(p for p in parts if p)
    return out


# ----------------------------- Main -----------------------------
def main():
    try:
        ap = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
        ap.add_argument("--control_config", required=True)
        ap.add_argument(
            "--trace_path",
            required=True,
            nargs="+",
            action="append",
            help="One or more root folders to watch; all **/*.jsonl under each will be included. "
                 "Can be specified multiple times for multiple data sources with fallback. "
                 "Example: --trace_path dir1 dir2 --trace_path dir3 (dir1,dir2 are primary; dir3 is fallback)",
        )
        ap.add_argument("--load_path", default=None)
        ap.add_argument("--delta-min", type=float, default=5e-4)
        ap.add_argument("--delta-max", type=float, default=5)
        ap.add_argument("--batch-rl", action = 'store_true')
        ap.add_argument("--reuse-traces", action='store_true',
                        help="After all traces are loaded once, re-use existing traces for offline testing")
        args = ap.parse_args()

        cfg_stem = Path(args.control_config).parent.stem
        with open(args.control_config, "r", encoding="utf-8") as f:
            control_cfg = json.load(f)

        # Registries
        pol_manifest = control_cfg["policy_cfg"]
        pol_name = pol_manifest["policy_name"]
        PolicyCfg = POLICY_CFG_REGISTRY[pol_name]
        PolicyCls = POLICY_REGISTRY[pol_name]
        roll_cfg = control_cfg["rollout_cfg"]
        BufferCls = BUFFER_REGISTRY[roll_cfg["buffer_name"]]

        # Trace watcher (multi-root, recursive *.jsonl)
        # Support multiple data sources with fallback
        # args.trace_path is now a list of lists: [[dir1, dir2], [dir3]]
        # Each inner list is a separate data source group
        trace_source_groups = [_normalize_trace_paths(group) for group in args.trace_path]
        watchers: List[TraceWatcher] = []
        for roots in trace_source_groups:
            watcher = TraceWatcher(roots, control_cfg, max_step=3)
            watchers.append(watcher)

        # Try to load initial traces from primary source first, then fallback sources
        init_traces, interference_vals = [], []
        for watcher in watchers:
            init_traces, interference_vals = watcher.load_initial_traces()
            if init_traces:
                print(f"[Data Source] Loaded {len(init_traces)} traces from source")
                break
        # If still empty, wait and retry
        while init_traces == []:
            for watcher in watchers:
                init_traces, interference_vals = watcher.load_initial_traces()
                if init_traces:
                    print(f"[Data Source] Loaded {len(init_traces)} traces from source after waiting")
                    break
            if not init_traces:
                time.sleep(1)

        # Build buffer
        writer = SummaryWriter(f"net_util/logs/{cfg_stem}")
        buf = BufferCls.build_from_traces(
            init_traces,
            device="cuda",
            advantage_estimator=roll_cfg.get("advantage_estimator"),
            gamma=pol_manifest.get("gamma"),
            lam=roll_cfg.get("lam"),
            reward_agg=roll_cfg.get("reward_agg", "sum"),
            buffer_max=roll_cfg.get("buffer_max", 10_000_000),
            interference_vals = interference_vals,
            writer = writer,
            capacity = 10000 if args.batch_rl else 20000,
        )

        # Policy
        pol_cfg = _inflate_dataclass_from_manifest(PolicyCfg, pol_manifest)
        if args.batch_rl:
            pol_cfg.batch_rl = True

        policy: PolicyBase = PolicyCls(ControlCmd, pol_cfg, rollout_buffer=buf, device="cuda", reward_cfg = control_cfg.get("reward_cfg"))

        # Checkpoint pathing
        if not args.load_path or str(args.load_path).lower() == "none":
            ckpt_dir = Path("net_util/net_cp") / cfg_stem
            next_id = 1
        else:
            print("Load model")
            policy.load(args.load_path, device="cuda")
            ckpt_dir = Path(args.load_path).parent
            next_id = int(Path(args.load_path).stem) + 1
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        store_path = ckpt_dir / f"{next_id}.pt"
        latest_path = ckpt_dir / f"latest.pt"
        # policy.load(latest_path, device="cuda")

        # TB + training
        actor_before = _flatten_params(policy.net.actor_parameters())

        def _extend_with_new():
            # Try to poll from primary watcher first, then fallback watchers
            new_traces, interference_vals = [], []

            for i, watcher in enumerate(watchers):
                new_traces, interference_vals = watcher.poll_new_traces()
                if new_traces:
                    print(f"[Data Source] Collected {len(new_traces)} traces from source {i}")
                    break

            # If no new traces and reuse is enabled, try to reload from any watcher
            if not new_traces and args.reuse_traces:
                for i, watcher in enumerate(watchers):
                    # Check if we have seen traces to reuse (after initial loading)
                    if len(watcher._seen) > 0:
                        print(f"[Trace Reuse] No new traces from source {i}, reusing existing traces...")
                        new_traces, interference_vals = watcher.reset_and_reload_all()
                        # Shuffle to vary the order
                        if new_traces:
                            import random
                            random.shuffle(new_traces)
                            random.shuffle(interference_vals)
                            print(f"[Trace Reuse] Reloaded {len(new_traces)} traces from source {i}")
                            break

            if new_traces != []:
                policy.buf.extend(
                    new_traces,
                    device="cuda",
                    reward_agg=roll_cfg.get("reward_agg", "sum"),
                    advantage_estimator=roll_cfg.get("advantage_estimator"),
                    gamma=pol_manifest.get("gamma"),
                    lam=roll_cfg.get("lam"),
                    interference_vals = interference_vals,
                )
                return True
            return False

        epoch = 0
        store_int = 50000
        last_trained_time = time.time()
        while True:
            no_need_update = policy.train_per_epoch(epoch, writer=writer, is_batch_rl=args.batch_rl)
            epoch += 1

            # Actor drift
            actor_after = _flatten_params(policy.net.actor_parameters())
            delta = float((actor_after - actor_before).norm(p=2).item())
            writer.add_scalar("params/delta_actor_epoch_L2", delta, epoch)

            # Big jump → save and roll checkpoint id
            trained_time = time.time()
            if delta >= args.delta_max or (trained_time - last_trained_time) >= 10:
                actor_before = actor_after
                last_trained_time = trained_time
                policy.save(latest_path)

            if epoch % store_int == 0: # For checkpointer purpose
                policy.save(store_path)
                next_id += 1
                store_path = ckpt_dir / f"{next_id}.pt"

            if not no_need_update:
                res = _extend_with_new()
                continue

    except Exception as e:
        # Save error to file
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }

        # Save to errors.jsonl file
        with open('errors.jsonl', 'a') as f:
            f.write(json.dumps(error_log) + '\n')

        print(f"Error saved to errors.jsonl: {error_log['error_type']}: {error_log['error_message']}")
        raise


if __name__ == "__main__":
    # Example: generate test config files then run
    # import importlib.util
    # cfg_path = Path("/home/qianren/workspace/dual_wifi_AP/config/rnn_test_2.py")
    # spec = importlib.util.spec_from_file_location("exp_config", cfg_path)
    # cfg_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg_mod)
    # policy_configs = cfg_mod.policy_config()

    # net_dir = Path("net_util/net_config/test_rnn_2"); net_dir.mkdir(parents=True, exist_ok=True)
    # with open(net_dir / "test.json", "w") as f:
    #     json.dump(next(iter(next(iter(policy_configs.values())).values())), f, indent=4)
    
    main()
