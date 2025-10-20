#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control + stochastic collector agent for the UDP IPC in util/ipc.py.

- PolicyBase initializes from a ControlCmd class type
- Infers action dimension from ControlCmd.__post_init__
- Provides cmd<->vector canonical mappings
"""

import argparse
import json
import random
import signal
import socket
import sys
import time
from dataclasses import dataclass, MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, Union

# --- Import your IPC wrapper (project root = one level up) ---
sys.path.append(str(Path(__file__).resolve().parents[1]))
from net_util.base import PolicyBase
from util.ipc import ipc_control  # noqa: E402
from util.control_cmd import ControlCmd, _json_default, revive_jsonlike  # noqa: E402
from util.trace_collec import trace_filter
from net_util import POLICY_REGISTRY, POLICY_CFG_REGISTRY

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

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

# -------------------------
# Agent config / plumbing
# -------------------------
@dataclass
class AgentConfig:
    server_ip: str
    server_port: int
    local_port: int
    links: List[str]
    sigma: float
    iterations: int
    period: float
    seed: Optional[int]
    use_nn: bool
    stats_retries: int
    fail_fast: bool
    out_dir: Path
    duration: Optional[float] = None  # seconds; if set, overrides iterations
    default_cmd: Optional[Dict] = None


class GracefulExit(Exception):
    pass


def _install_signal_handlers():
    def handler(signum, frame):
        raise GracefulExit()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# --------------------------------
# IPC wrappers with robust retry
# --------------------------------
def ipc_get_statistics(ctrl: ipc_control,
                       retries: int,
                       duration: float,
                       backoff0: float = 0.5,
                       fail_fast: bool = False,
                       first_ok_stats_t = None,
                       ) -> Optional[Dict[str, Any]]:
    attempt = 0
    delay = backoff0
    while True:
        try:
            return ctrl.statistics()
        except socket.timeout:
            attempt += 1
            if attempt > retries or fail_fast:
                return None
            time.sleep(delay)
            if first_ok_stats_t is not None:
                elapsed = time.time() - first_ok_stats_t
                print(elapsed, delay, duration)
                if elapsed + delay > duration:
                    return None
            delay *= 1.6
        except OSError:
            attempt += 1
            if attempt > retries or fail_fast:
                return None
            time.sleep(delay)
            delay *= 1.6

# -------------------------
# Main agent loop
# -------------------------
def run_agent(cfg: AgentConfig, policy: PolicyBase, state_cfg: Dict, is_eval: bool):
    ensure_dir(cfg.out_dir)
    jsonl_path = cfg.out_dir / "rollout.jsonl"

    _install_signal_handlers()

    # Build policy with knowledge of ControlCmd structure
    action_dim = policy.action_dim
    print(f"[INFO] Inferred action dimension from ControlCmd: {action_dim}")

    ctrl = ipc_control(cfg.server_ip, cfg.server_port, cfg.local_port, link_name="agent")

    last_res = None
    default_res = {
        "action": [0, 0, 0, 0, 0], "log_prob": [0], "value": 0
    }
    # Duration control
    first_ok_stats_t: Optional[float] = None
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        print(f"[INFO] Agent started. Logging to {cfg.out_dir}")
        next_time = time.monotonic()

        # Keep iteration counting for logs, but duration can stop earlier
        for it in range(cfg.iterations):
            # 1) Observe (with retries/backoff)
            stats = ipc_get_statistics(
                ctrl, retries=cfg.stats_retries, backoff0=0.5, fail_fast=cfg.fail_fast, first_ok_stats_t = first_ok_stats_t, duration = cfg.duration
            )
            timed_out = stats is None

            # Start duration timer at first successful stats
            if not timed_out and first_ok_stats_t is None:
                first_ok_stats_t = time.time()
                print(first_ok_stats_t)
                if cfg.duration is not None:
                    print(f"[INFO] Duration timer started at t={first_ok_stats_t:.3f} "
                          f"(max {cfg.duration:.3f}s).")

            # Enforce duration *immediately after* stats collection (higher priority than iterations)
            if cfg.duration is not None and first_ok_stats_t is not None:
                elapsed = time.time() - first_ok_stats_t
                if elapsed >= cfg.duration:
                    print(f"[INFO] Duration reached: elapsed={elapsed:.3f}s ≥ {cfg.duration:.3f}s. "
                          f"Stopping before sending further actions.")
                    break
            
            if timed_out:
                continue
            
            obs_for_policy = {} if timed_out else trace_filter(
                {'stats': stats, 'res': last_res if last_res is not None else default_res}
                , state_cfg
            )
            # 2) Base action + stochastic exploration
            if cfg.default_cmd is not None:
                control_cmd:ControlCmd = revive_jsonlike(cfg.default_cmd)
                res = None
            else:
                try:
                    res, control_cmd = policy.act(obs_for_policy, is_eval)
                    last_res = res
                except Exception as e:
                    print(e)
                    continue
            # Build ControlCmd using canonical mapping (pads/trims internally)
            control_body: Dict[str, ControlCmd] = {}
            ## TODO: handle multi-link control
            for link in cfg.links:
                try:
                    control_body[link] = control_cmd
                except (TypeError, ValueError) as e:
                    print(f"[ERROR] Failed to build ControlCmd for {link}: {e}", file=sys.stderr)

            # 3) Transmit control
            try:
                if control_body:
                    ctrl.control(control_body)
            except OSError as e:
                print(f"[WARN] send control failed: {e}", file=sys.stderr)

            # 5) Log
            t_now = time.time()
            record = {
                "t": t_now,
                "iteration": it,
                "links": cfg.links,
                "action": control_body,  # final cmd per link
                "stats": stats,
                "timed_out": timed_out,
                "res": res,
                "policy": policy.__class__.__name__,
            }
            jf.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            # 6) Pace loop
            next_time += cfg.period
            sleep_for = next_time - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_time = time.monotonic()

    try:
        ctrl.release()
    except Exception:
        pass
    print("[INFO] Agent stopped gracefully.")

def parse_args() -> Tuple[AgentConfig, PolicyBase]:
    p = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
    p.add_argument("--control_config", type=str, required=True)
    p.add_argument("--transmission_config", type=str, required=True)
    p.add_argument("--eval", action="store_true")
    
    args = p.parse_args()
    
    with open(args.transmission_config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        
    links = []
    for stream in config_data["streams"]:
        name = str(stream.get("port")) + "@" + str(stream.get("tos"))
        links.append(name)
        
    with open(args.control_config, 'r', encoding='utf-8') as f:
        control_config = json.load(f)

    agent_cfg = control_config['agent_cfg']
    agent_cfg["links"] = links
    agent_cfg["iterations"] = agent_cfg["duration"] // agent_cfg["period"]
    
    cfg = _inflate_dataclass_from_manifest(AgentConfig, agent_cfg)
    
    policy_cfg = control_config['policy_cfg']
    policy_name = policy_cfg["policy_name"]
    policy_cfg_cls = POLICY_CFG_REGISTRY[policy_name]
    policy_cls = POLICY_REGISTRY[policy_name]
    
    policy_construct_cfg = _inflate_dataclass_from_manifest(policy_cfg_cls, policy_cfg)
    policy: PolicyBase = policy_cls( ControlCmd, policy_construct_cfg, state_transform_dict = control_config['state_transform_dict'])
    
    policy_load_path = policy_cfg['load_path']
    policy_cp_path = Path("net_util/net_cp") / Path(args.control_config).parent.stem
    if policy_load_path == 'latest':
        ids = [int(p.stem) for p in policy_cp_path.glob("*.pt") if p.stem.isdigit()]
        if not ids:
            policy_load_path = None
        else:
            policy_load_path = policy_cp_path / f"{max(ids)}.pt"
    else:
        policy_load_path = policy_cp_path / policy_load_path
        
    if policy_load_path:
        print(f"PPO Load {policy_load_path}")
        try:
            policy.load(policy_load_path, device=policy_cfg['device'])
        except:
            print("Load fail; Fall back to no model")
    state_cfg = control_config.get('state_cfg', None)    
    return cfg, policy, state_cfg, args.eval
    
def main():
    try:
        cfg, policy, state_cfg, is_eval = parse_args()
        run_agent(cfg, policy, state_cfg, is_eval)
    except GracefulExit:
        print("\n[INFO] Caught interrupt, exiting.")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, exiting.")


if __name__ == "__main__":
    main()
