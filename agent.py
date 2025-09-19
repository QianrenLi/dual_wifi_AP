#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control + stochastic collector agent for the UDP IPC in util/ipc.py.

- PolicyBase initializes from a ControlCmd class type
- Infers action dimension from ControlCmd.__post_init__
- Provides cmd<->vector canonical mappings
"""

import argparse
import csv
import json
import random
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import numpy as np

# --- Import your IPC wrapper (project root = one level up) ---
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.ipc import ipc_control  # noqa: E402
from util.control_cmd import ControlCmd, list_to_cmd, _json_default  # noqa: E402

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    
def to_len_n_float_list(values: List[float], n: int) -> List[float]:
    """Cast to float and pad/trim to exactly n."""
    arr = [float(x) for x in values]
    if len(arr) < n:
        arr += [0.0] * (n - len(arr))
    elif len(arr) > n:
        arr = arr[:n]
    return arr

# -------------------------
# Policy / NN placeholders
# -------------------------
class PolicyBase:
    """
    Base policy that understands the action structure via ControlCmd class.
    Provides canonical mappings:
      - cmd_to_vec(ControlCmd) -> List[float]
      - vec_to_cmd(List[float], version) -> ControlCmd
    """

    def __init__(self, cmd_cls: Type[ControlCmd], seed: Optional[int] = None):
        self.cmd_cls = cmd_cls
        self.action_dim: int = cmd_cls.__dim__()
        if seed is not None:
            random.seed(seed)

    # --- policy API ---
    def act(self, obs: Dict[str, Any]) -> List[float]:
        """Deterministic base action (vector of length action_dim)."""
        raise NotImplementedError


class ZeroPolicy(PolicyBase):
    def act(self, obs: Dict[str, Any]) -> List[float]:
        return [0.0] * self.action_dim


class TorchMLPPolicy(PolicyBase):
    """
    Tiny MLP (if PyTorch available). Maps a numeric obs vector to action_dim.
    """
    def __init__(self, cmd_cls: Type[ControlCmd], seed: Optional[int] = None, hidden: int = 32):
        super().__init__(cmd_cls, seed)
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
        except Exception as e:
            raise RuntimeError("PyTorch not available; omit --use-nn or install torch.") from e

        if seed is not None:
            self.torch.manual_seed(seed)

        self.model = None
        self.hidden = hidden

    def _build(self, in_dim: int):
        self.model = self.nn.Sequential(
            self.nn.Linear(in_dim, self.hidden),
            self.nn.ReLU(),
            self.nn.Linear(self.hidden, self.action_dim),
            self.nn.Tanh(),
        )

    def _obs_to_vec(self, obs: Dict[str, Any]) -> List[float]:
        vec: List[float] = []
        def push(v):
            if isinstance(v, (int, float)):
                vec.append(float(v))
        for _, v in sorted(obs.items()):
            if isinstance(v, dict):
                for __, vv in sorted(v.items()):
                    push(vv)
            else:
                push(v)
        return vec or [0.0]

    def act(self, obs: Dict[str, Any]) -> List[float]:
        import torch
        x = self._obs_to_vec(obs)
        if self.model is None:
            self._build(len(x))
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = self.model(t)
        return y.squeeze(0).tolist()


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
def run_agent(cfg: AgentConfig):
    ensure_dir(cfg.out_dir)
    jsonl_path = cfg.out_dir / "rollout.jsonl"

    rng = random.Random(cfg.seed)
    _install_signal_handlers()

    # Build policy with knowledge of ControlCmd structure
    if cfg.use_nn:
        try:
            policy = TorchMLPPolicy(ControlCmd, seed=cfg.seed)
        except RuntimeError as e:
            print(f"[WARN] {e} Falling back to ZeroPolicy.", file=sys.stderr)
            policy = ZeroPolicy(ControlCmd, seed=cfg.seed)
    else:
        policy = ZeroPolicy(ControlCmd, seed=cfg.seed)

    action_dim = policy.action_dim
    print(f"[INFO] Inferred action dimension from ControlCmd: {action_dim}")

    ctrl = ipc_control(cfg.server_ip, cfg.server_port, cfg.local_port, link_name="agent")

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
                print(elapsed)
                if elapsed >= cfg.duration:
                    print(f"[INFO] Duration reached: elapsed={elapsed:.3f}s ≥ {cfg.duration:.3f}s. "
                          f"Stopping before sending further actions.")
                    break
            
            if timed_out:
                continue
            
            obs_for_policy = {} if timed_out else stats

            # 2) Base action + stochastic exploration
            base_vec = policy.act(obs_for_policy)
            noisy_vec = [float(a) + rng.gauss(0.0, float(cfg.sigma)) for a in base_vec]

            # Build ControlCmd using canonical mapping (pads/trims internally)
            control_body: Dict[str, ControlCmd] = {}
            for link in cfg.links:
                try:
                    cmd = list_to_cmd(policy.cmd_cls, noisy_vec)
                    control_body[link] = cmd
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



# -------------------------
# CLI
# -------------------------
def parse_args() -> AgentConfig:
    p = argparse.ArgumentParser(description="Control + stochastic collect agent (ControlCmd-aware)")
    p.add_argument("--server-ip", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=11112)
    p.add_argument("--local-port", type=int, default=12345)
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Stddev of exploration noise added to base action")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--period", type=float, default=0.5,
                   help="Seconds between control steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use-nn", action="store_true",
                   help="Try a tiny Torch MLP (falls back to zeros if torch missing)")
    p.add_argument("--stats-retries", type=int, default=3,
                   help="Retries for statistics() on timeout")
    p.add_argument("--fail-fast", action="store_true",
                   help="If set, exit statistics() immediately on first timeout")
    p.add_argument("--out-dir", type=str, default="logs/agent",
                   help="Directory for JSONL/CSV logs")
    p.add_argument("--duration", type=float, default=None,
                   help="Max runtime in seconds, counted from FIRST successful stats collection. "
                        "If set, it overrides --iterations.")
    p.add_argument("--config", type=str, required= True)
    args = p.parse_args()
    
    if args.duration is not None:
        args.iterations = int(args.duration / args.period) + 1
    
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    links = []
    for stream in config_data["streams"]:
        name = str(stream.get("port")) + "@" + str(stream.get("tos"))
        links.append(name)
    
    return AgentConfig(
        server_ip=args.server_ip,
        server_port=args.server_port,
        local_port=args.local_port,
        links=links,
        sigma=args.sigma,
        iterations=args.iterations,
        period=args.period,
        seed=args.seed,
        use_nn=args.use_nn,
        stats_retries=args.stats_retries,
        fail_fast=args.fail_fast,
        out_dir=Path(args.out_dir),
        duration = args.duration,
    )


def main():
    try:
        cfg = parse_args()
        run_agent(cfg)
    except GracefulExit:
        print("\n[INFO] Caught interrupt, exiting.")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, exiting.")


if __name__ == "__main__":
    main()
