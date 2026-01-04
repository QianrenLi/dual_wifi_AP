# train_rl.py
import argparse, json, os, time
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin
from datetime import datetime
import traceback
from multiprocessing import Process, Queue

import torch as th
from torch.utils.tensorboard import SummaryWriter

from net_util import POLICY_REGISTRY, POLICY_CFG_REGISTRY, BUFFER_REGISTRY
from net_util.base import PolicyBase
# Ensure all algorithms are imported
from util.control_cmd import ControlCmd
from util.trace_watcher import TraceWatcher
from typing import List

#------------------------------ Constants ---------------------------
EPISOID_LENGTH = 350

# TraceWatcher method names
METHOD_POLL_NEW_TRACES = "poll_new_traces"
METHOD_LOAD_INITIAL_TRACES = "load_initial_traces"
METHOD_RESET_AND_RELOAD_ALL = "reset_and_reload_all"


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


# ----------------------------- Data Loader Worker -----------------------------
def _data_loader_worker(watchers, queue, stop_event, roll_cfg, pol_manifest):
    """
    Background process that continuously loads and preprocesses traces.
    Performs ALL CPU-intensive preprocessing:
    - JSON parsing and trace collection
    - State transformation
    - _tracify() conversion to numpy arrays
    - Episode chunking
    - Return calculation
    - Reward/gamma summaries

    Puts pre-processed Episode objects into the queue for fast buffer insertion.
    """
    # Import preprocessing functions - these are CPU-intensive
    from net_util.replay.rnn_fifo import (
        _tracify,
        Episode
    )
    import numpy as np

    gamma = pol_manifest.get("gamma", 0.99)
    episode_length = roll_cfg.get("episode_length", EPISOID_LENGTH)
    reward_agg = roll_cfg.get("reward_agg", "sum")
    advantage_estimator = roll_cfg.get("advantage_estimator")

    try:
        while not stop_event.is_set():
            new_traces, interference_vals = [], []
            source_idx = -1  # Track which watcher provided the data

            # Try to poll from primary watcher first, then randomly from fallback watchers
            new_traces, interference_vals, source_idx = _poll_from_watchers(watchers, METHOD_POLL_NEW_TRACES)

            # If we have new traces, preprocess them fully
            if new_traces:
                preprocessed_episodes = []

                for (states, actions, rewards, network_output), interf in zip(new_traces, interference_vals):
                    # Step 1: Convert to numpy arrays (was _tracify in buf.extend)
                    obs_np, act_np, rew_np, next_obs_np, done_np = _tracify(
                        states, actions, rewards, network_output, reward_agg
                    )

                    # Step 2: Split into episodes of fixed length
                    num_episodes = len(obs_np) // episode_length
                    for i in range(num_episodes):
                        start = i * episode_length
                        end = (i + 1) * episode_length

                        # Step 3: Create Episode with all preprocessing done
                        # This includes: reward_summary, gamma_summary, return calculation
                        ep = Episode(
                            obs_np[start:end],
                            act_np[start:end],
                            rew_np[start:end],
                            next_obs_np[start:end],
                            done_np[start:end],
                            init_loss=10000.0,
                            gamma=gamma,
                            interference=interf
                        )
                        preprocessed_episodes.append(ep)

                # Put preprocessed episodes and source info in queue
                # First watcher (index 0) is "current", others are "background"
                if preprocessed_episodes:
                    source = "current" if source_idx == 0 else "background"
                    queue.put((preprocessed_episodes, source))
            else:
                # No new data, sleep briefly
                time.sleep(0.1)
    except Exception as e:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'process': 'data_loader_worker',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        with open('errors.jsonl', 'a') as f:
            f.write(json.dumps(error_log) + '\n')
        print(f"[DataLoader] Error: {e}")


# ----------------------------- Helper Functions -----------------------------
def _poll_from_watchers(watchers, method_name=METHOD_POLL_NEW_TRACES, shuffle_result=False):
    """
    Poll from watchers: try primary first, then randomly try fallback watchers.

    Args:
        watchers: List of TraceWatcher objects
        method_name: Name of the method to call on each watcher
        shuffle_result: If True, shuffle the returned traces

    Returns:
        tuple: (traces, interference_vals, source_idx)
    """
    import random

    # Try primary watcher first
    method = getattr(watchers[0], method_name)
    new_traces, interference_vals = method()
    source_idx = 0 if new_traces else -1

    # If primary has no data, randomly try fallback watchers
    if not new_traces and len(watchers) > 1:
        fallback_indices = list(range(1, len(watchers)))
        random.shuffle(fallback_indices)
        for i in fallback_indices:
            method = getattr(watchers[i], method_name)
            new_traces, interference_vals = method()
            if new_traces:
                source_idx = i
                break

    # Shuffle results if requested (for reuse_traces)
    if shuffle_result and new_traces:
        random.shuffle(new_traces)
        random.shuffle(interference_vals)

    return new_traces, interference_vals, source_idx


def _reload_from_watchers(watchers):
    """
    Reload from watchers: try primary first, then randomly try fallback watchers.
    Only tries watchers that have seen traces (len(_seen) > 0).

    Returns:
        tuple: (traces, interference_vals, source_idx)
    """
    import random

    # Try primary watcher first
    if len(watchers[0]._seen) > 0:
        new_traces, interference_vals = getattr(watchers[0], METHOD_RESET_AND_RELOAD_ALL)()
        source_idx = 0 if new_traces else -1
    else:
        new_traces, interference_vals = [], []
        source_idx = -1

    # If primary has no data, randomly try fallback watchers
    if not new_traces and len(watchers) > 1:
        fallback_indices = list(range(1, len(watchers)))
        random.shuffle(fallback_indices)
        for i in fallback_indices:
            if len(watchers[i]._seen) > 0:
                new_traces, interference_vals = getattr(watchers[i], METHOD_RESET_AND_RELOAD_ALL)()
                if new_traces:
                    source_idx = i
                    # Shuffle to vary the order
                    random.shuffle(new_traces)
                    random.shuffle(interference_vals)
                    break

    return new_traces, interference_vals, source_idx


# ----------------------------- Main -----------------------------
def main():
    try:
        ap = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
        ap.add_argument("--control_config", required=True)
        ap.add_argument(
            "--trace_path",
            required=True,
            nargs="+",
            help="Root folders to watch with fallback. First path is primary source, "
                 "remaining paths are fallback sources when primary is empty. "
                 "Example: --trace_path primary_dir fallback_dir1 fallback_dir2",
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
        # Support multiple data sources with fallback.
        # First path is primary, remaining paths are fallbacks.
        # Each path gets its own TraceWatcher.
        trace_paths = _normalize_trace_paths(args.trace_path)
        watchers: List[TraceWatcher] = []
        for path in trace_paths:
            watcher = TraceWatcher([path], control_cfg, max_step=3)
            watchers.append(watcher)

        # Try to load initial traces from primary source first, then randomly from fallback sources
        init_traces, interference_vals, _ = _poll_from_watchers(watchers, METHOD_LOAD_INITIAL_TRACES)

        # If still empty, wait and retry
        while init_traces == []:
            init_traces, interference_vals, _ = _poll_from_watchers(watchers, METHOD_LOAD_INITIAL_TRACES)
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
            episode_length = roll_cfg.get("episode_length", EPISOID_LENGTH),
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
            # Try to poll from primary watcher first, then randomly from fallback watchers
            new_traces, interference_vals, source_idx = _poll_from_watchers(watchers, METHOD_POLL_NEW_TRACES)

            # If no new traces and reuse is enabled, try to reload from any watcher
            if not new_traces and args.reuse_traces:
                new_traces, interference_vals, source_idx = _reload_from_watchers(watchers)

            if new_traces != []:
                # First watcher (index 0) is "current", others are "background"
                source = "current" if source_idx == 0 else "background"
                policy.buf.extend(
                    new_traces,
                    device="cuda",
                    reward_agg=roll_cfg.get("reward_agg", "sum"),
                    advantage_estimator=roll_cfg.get("advantage_estimator"),
                    gamma=pol_manifest.get("gamma"),
                    lam=roll_cfg.get("lam"),
                    interference_vals=interference_vals,
                    source=source,  # Pass source parameter
                )
                return True
            return False

        epoch = 0
        store_int = 50000
        last_trained_time = time.time()

        # Setup multiprocessing for data loading
        data_queue = Queue(maxsize=5)  # Buffer up to 5 batches
        stop_event = None  # We'll create this conditionally
        loader_proc = None

        # Only start background worker if not in batch-rl mode
        if not args.batch_rl:
            from multiprocessing import Event
            stop_event = Event()
            loader_proc = Process(
                target=_data_loader_worker,
                args=(watchers, data_queue, stop_event, roll_cfg, pol_manifest)
            )
            loader_proc.start()
            print(f"[Main] Started background data loader (PID: {loader_proc.pid})")

        try:
            while True:
                no_need_update = policy.train_per_epoch(epoch, writer=writer, is_batch_rl=args.batch_rl)
                epoch += 1

                # Big jump → save and roll checkpoint id
                trained_time = time.time()
                if trained_time - last_trained_time >= 120:
                    last_trained_time = trained_time
                    policy.save(latest_path)

                if epoch % store_int == 0:
                    policy.save(store_path)
                    next_id += 1
                    store_path = ckpt_dir / f"{next_id}.pt"

                # Load new data (non-blocking from queue or synchronous for batch-rl)
                if not no_need_update:
                    if not args.batch_rl:
                        # Try to get preprocessed episodes from queue without blocking
                        try:
                            result = data_queue.get_nowait()
                            if result:
                                # Unpack (preprocessed_episodes, source) tuple
                                preprocessed_episodes, source = result
                                # Fast insertion - all CPU work already done in background
                                policy.buf.add_preprocessed_episodes(preprocessed_episodes, source=source)
                                # print(f"[Main] Added {len(preprocessed_episodes)} preprocessed episodes (source={source}) to buffer")
                        except:
                            # Queue is empty, will try again next epoch
                            pass
                    else:
                        # Batch-RL mode: use original synchronous loading
                        res = _extend_with_new()
        finally:
            # Cleanup background process
            if not args.batch_rl and stop_event is not None and loader_proc is not None:
                stop_event.set()
                loader_proc.join(timeout=5)
                if loader_proc.is_alive():
                    loader_proc.terminate()
                print("[Main] Background data loader stopped")

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
