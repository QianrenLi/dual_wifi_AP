#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from tap import Connector
from util.exp_setup import create_transmission_config
from util.flows import flow_to_rtt_log, Flow

TxSrcs = Dict[str, Dict[str, str]]        # tx -> {"control_config": "...", "transmission_config": "..."}
Flows = Dict[int, Flow]                   # port -> Flow

# ---------- Low-level utilities ----------
import random
from typing import Tuple, Type

def apply_until_done(
    conn: "Connector",
    pause_s: float = 1.0,
    *,
    timeout_s: float | None = None,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    backoff: float = 1.0,      # 1.0 = constant delay; >1.0 = exponential backoff
    jitter_s: float = 0.0,     # add up to ±jitter_s/2 random jitter to sleep
) -> any:
    """
    Flush the executor and keep applying until it succeeds, or until timeout.

    Args:
        conn: Connector instance.
        pause_s: Base pause between retries (seconds).
        timeout_s: Maximum total time to keep retrying (seconds). None = no timeout.
        retry_exceptions: Which exceptions should trigger a retry.
        backoff: Multiplier for exponential backoff (1.0 to disable).
        jitter_s: Uniform jitter range added to sleep (0 to disable).

    Returns:
        Whatever `conn.apply()` returns on success.

    Raises:
        TimeoutError: If `timeout_s` elapses without success.
        Any non-retry exception from `conn.apply()` is raised immediately.
    """
    start = time.monotonic()
    conn.executor.fetch()
    attempt = 0

    while True:
        attempt += 1
        try:
            return conn.apply()
        except retry_exceptions as e:
            # Non-infinite timeout check
            if timeout_s is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout_s:
                    raise TimeoutError(
                        f"apply_until_done timed out after {elapsed:.2f}s "
                        f"and {attempt} attempt(s)."
                    ) from e

            # Compute sleep with optional backoff and jitter
            sleep_s = pause_s * (backoff ** (attempt - 1))
            if jitter_s > 0:
                sleep_s += random.uniform(-jitter_s / 2, jitter_s / 2)

            # Don't oversleep past the deadline
            if timeout_s is not None:
                remaining = timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError(
                        f"apply_until_done timed out after {timeout_s:.2f}s "
                        f"and {attempt} attempt(s)."
                    ) from e
                sleep_s = max(0.0, min(sleep_s, remaining))

            time.sleep(sleep_s)


def rel_config_path(path_str: str) -> str:
    """The remote side expects path without the leading first segment."""
    # original code: "/".join(srcs['transmission_config'].split("/")[1:])
    parts = Path(path_str).as_posix().split("/")
    return "/".join(parts[1:]) if len(parts) > 1 else parts[0]


def now_trial_folder(exp_name: str, interference_level: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(f"exp_trace/{exp_name}/IL_{interference_level}_trial_{ts}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def clean_first_iteration(exp_name: str):
    """Only used at iteration 0 to start fresh."""
    for p in (Path(f"exp_trace/{exp_name}"), Path(f"net_util/net_cp/{exp_name}")):
        if p.exists():
            print(f"[clean] removing {p}")
            shutil.rmtree(p, ignore_errors=True)


def schedule_receive(conn: Connector, flow: Flow, port: int, duration: int, render: bool, wait: float = 0.0):
    """Queue the appropriate receive command for one flow."""
    needs_file = "file" in flow.npy_file
    base_args = {"duration": duration, "port": port, "timeout": duration + 20, "hyper_parameters":""} #[Warning]: it seems the tap have certain memory so the hyper parameter should modified each time
    if needs_file:
        conn.batch(flow.dst_sta, "receive_file", base_args).wait(wait)
        return

    # with RTT calculation and first tx IP
    ip = flow.tx_ipaddrs[0]
    hp = f"--calc-rtt --src-ipaddrs {ip}"
    if render:
        # note: original adds '--rx-mode' only for GUI
        conn.batch(flow.dst_sta, "receive_file_gui", {**base_args, "hyper_parameters": f"{hp} --rx-mode"}).wait(wait)
    else:
        conn.batch(flow.dst_sta, "receive_file", {**base_args, "hyper_parameters": hp}).wait(wait)


def schedule_send(conn: Connector, tx_srcs: TxSrcs, duration: int, *, cmd: str):
    """Queue send commands for a set of tx sources."""
    for tx, srcs in tx_srcs.items():
        cfg_rel = rel_config_path(srcs["transmission_config"])
        conn.batch(tx, cmd, {"duration": duration, "config": cfg_rel, "timeout": duration + 20})


def pull_rtt_logs(folder: Path, tx_flows: Flows, log_dir: Path = Path("stream-replay/logs")):
    """Pull per-flow RTT logs from clients and move into the trial folder."""
    for flow in tx_flows.values():
        src_client = flow.src_sta
        fname = flow_to_rtt_log(flow)
        Connector(src_client).sync_file(str(log_dir / fname))
        (folder / fname).write_bytes((log_dir / fname).read_bytes())
        (log_dir / fname).unlink(missing_ok=True)


def move_rollout_from_tx(folder: Path, tx_srcs: TxSrcs):
    """Move logs/agent/rollout.jsonl into this trial folder (from any tx host)."""
    # original code just renames local file without an explicit sync per tx
    # preserve that behavior
    rollout = Path("logs/agent/rollout.jsonl")
    for tx in tx_srcs.keys():
        Connector(tx).sync_file("logs/agent/rollout.jsonl")
        if rollout.exists():
            rollout.rename(folder / "rollout.jsonl")
            break
    
    # output_log = Path("stream-replay/log/output.log")
    # for tx in tx_srcs.keys():
    #     Connector(tx).sync_file("stream-replay/log/output.log")
    #     if output_log.exists():
    #         output_log.rename(folder / "output.log")
    #         break


def push_rollout_to_trainer(folder: Path):
    """Send the rollout file to TrainAgent."""
    target = folder / "rollout.jsonl"
    if target.exists():
        Connector("TrainAgent").sync_file(str(target), is_pull=False)


def train_forever(conn: Connector, control_config: str, trace: Path, maybe_load: Path | None = None):
    # Connector("TrainAgent").killproc("train_rl.py", signal="-TERM")
    # print("clean up")
    # time.sleep(2)
    args = {"control_config": control_config, "trace_path": trace}
    if maybe_load is not None:
        args["load_path"] = str(maybe_load)
    conn.batch("TrainAgent", "model_train_forever", args)
    apply_until_done(conn)


def pull_trainer_artifacts(exp_name: str):
    """Pull model and train.log from TrainAgent; keep a copy in trial folder."""
    Connector("TrainAgent").sync_file(f"net_util/net_cp/{exp_name}/latest.pt", is_pull=True)

# ---------- High-level routines ----------

def start_agents(conn: Connector, tx_srcs: TxSrcs, eval_mode: bool = False):
    action = "start_agent_eval" if eval_mode else "start_agent"
    for tx, srcs in tx_srcs.items():
        conn.batch(tx, action, {"control_config": srcs["control_config"],
                                "transmission_config": srcs["transmission_config"]})


def schedule_all_receives(conn: Connector, tx_flows: Flows, inter_flows: Flows, duration: int, render: bool):
    # interference first (no waits)
    for port, flow in inter_flows.items():
        schedule_receive(conn, flow, port, duration, render=False, wait=0.0)

    # training/eval flows; wait on the last one
    ports = list(tx_flows.keys())
    for idx, port in enumerate(ports):
        flow = tx_flows[port]
        is_last = (idx == len(ports) - 1)
        schedule_receive(conn, flow, port, duration, render=render, wait=3.0 if is_last else 0.1)


def run_tx_and_interference(conn: Connector, tx_srcs: TxSrcs, inter_srcs: TxSrcs, duration: int):
    schedule_send(conn, tx_srcs, duration, cmd="send_file")
    schedule_send(conn, inter_srcs, duration, cmd="send_file_with_out_mon")


def run_iteration(
    args: argparse.Namespace,
    conn: Connector,
    tx_srcs: TxSrcs,
    tx_flows: Flows,
    inter_srcs: TxSrcs,
    inter_flows: Flows,
    duration: int,
    exp_name: str,
    iteration: int,
    traces: List[Path],
    max_traces: int,
    int_level: int = 0,
):
    start_t = time.time()

    # 1) start agents
    start_agents(conn, tx_srcs, eval_mode=args.evaluate)

    # 2) schedule receives & sends
    schedule_all_receives(conn, tx_flows, inter_flows, duration, render=args.render)
    run_tx_and_interference(conn, tx_srcs, inter_srcs, duration)

    # 3) run remote work
    try:
        res = apply_until_done(conn, pause_s=0.5, timeout_s=duration+30, backoff=1.5, jitter_s=0.2)
    except KeyboardInterrupt:
        exit()
    except:
        print("fail and retry")
        return iteration
    
    res = [r for r in res if r]  # drop {}
    exp_t = time.time()

    # 4) collect artifacts
    folder = now_trial_folder(exp_name, interference_level=int_level)
    pull_rtt_logs(folder, tx_flows)
    move_rollout_from_tx(folder, tx_srcs)
    
    # 5) training (only in train mode)
    if not args.evaluate:
        push_rollout_to_trainer(folder) 
        sync_t = time.time()
        
        traces.append(folder / "rollout.jsonl")
        while len(traces) > max_traces:
            traces.pop(0)
            
        # 6) pull new model & logs
        if iteration > 2:
            pull_trainer_artifacts(exp_name)
            
        end_t = time.time()

        print(f"Iteration {iteration}:")
        print(f"  Exp time   : {exp_t - start_t:.3f}s")
        print(f"  Sync time  : {sync_t - exp_t:.3f}s")
        # print(f"  Train time : {train_t - sync_t:.3f}s")
        print(f"  Pull time  : {end_t - sync_t:.3f}s")
        print(f"  Total      : {end_t - start_t:.3f}s")
    print(f"  res        : {res}")

    return iteration + (0 if args.evaluate else 1)


# ---------- Entry points ----------

def train_loop(
    args: argparse.Namespace,
    conn: Connector,
    paths: List,
    duration: int,
    exp_name: str,
):
    def path_id(iteration):
        # return (iteration // args.per_exp_trials) % len(paths)
        return random.randint(0, len(paths) - 1)
        
    traces: List[Path] = []
    max_traces = 1
    iteration = 0
    last_iteration = iteration
    
    interference_level = path_id(iteration)
    path = paths[ interference_level ]
    print(f"Train config {path}")
    tx_srcs, tx_flows, inter_srcs, inter_flows = create_transmission_config( path, exp_name, conn, is_update=True)
    
    if iteration == 0:
        clean_first_iteration(exp_name)

    any_tx = next(iter(tx_srcs))
    control_cfg = tx_srcs[any_tx]["control_config"]
    train_forever(conn, control_cfg, f'exp_trace/{exp_name}', None)
    
    while True:
        if iteration % args.per_exp_trials == 0:
            interference_level = path_id(iteration)
            path = paths[ interference_level ]
            print(f"Train config {path}")
            tx_srcs, tx_flows, inter_srcs, inter_flows = create_transmission_config( path, exp_name, conn, is_update=True)
            
        iteration = run_iteration(
            args, conn, tx_srcs, tx_flows, inter_srcs, inter_flows,
            duration, exp_name, iteration, traces, max_traces, int_level=interference_level
        )


def evaluate(
    args: argparse.Namespace,
    conn: Connector,
    paths: List,
    duration: int,
    exp_name: str,
):
    # single evaluation pass reusing the same building blocks
    for int_level, path in enumerate(paths): 
        print(f"Evaluate config {path}")
        tx_srcs, tx_flows, inter_srcs, inter_flows = create_transmission_config( path, exp_name, conn, is_update=True)
        _ = run_iteration(
            args, conn, tx_srcs, tx_flows, inter_srcs, inter_flows,
            duration, exp_name, iteration=0, traces=[], max_traces=1, int_level=int_level
            )


def main():
    parser = argparse.ArgumentParser(description="Run the transmission experiment.")
    parser.add_argument("--duration", type=int, default=5, help="Duration (seconds).")
    parser.add_argument("--exp_folder", type=str, default="", help="Experiment name.")
    parser.add_argument("--exp_name", type=str, default="local_exp", help="Experiment name.")
    parser.add_argument("--per_exp_trials", type=int, default=50, help="The iteration number of per experiment")
    parser.add_argument("--render", action="store_true", help="Render RX GUI.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluation mode (no training).")
    args = parser.parse_args()

    conn = Connector()
    
    if args.exp_folder != "":
        folder = Path(f'{args.exp_folder}')
        paths = sorted(folder.rglob("*.py"), key=lambda p: int(p.stem) )
    else:
        paths = [ Path("config") / f'{args.exp_name}.py' ]
    
    if args.evaluate:
        evaluate(args, conn, paths, args.duration, args.exp_name)
    else:
        train_loop(args, conn, paths, args.duration, args.exp_name)


if __name__ == "__main__":
    main()
