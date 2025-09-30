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
def apply_until_done(conn: Connector, pause_s: float = 1.0):
    """Flush the executor and keep applying until it succeeds."""
    conn.executor.fetch()
    while True:
        try:
            return conn.apply()
        except Exception:
            time.sleep(pause_s)


def rel_config_path(path_str: str) -> str:
    """The remote side expects path without the leading first segment."""
    # original code: "/".join(srcs['transmission_config'].split("/")[1:])
    parts = Path(path_str).as_posix().split("/")
    return "/".join(parts[1:]) if len(parts) > 1 else parts[0]


def now_trial_folder(exp_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(f"exp_trace/{exp_name}/trial_{ts}")
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
    base_args = {"duration": duration, "port": port, "hyper_parameters":""} #[Warning]: it seems the tap have certain memory so the hyper parameter should modified each time
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
        conn.batch(tx, cmd, {"duration": duration, "config": cfg_rel})


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
    if rollout.exists():
        rollout.rename(folder / "rollout.jsonl")
    else:
        # if not present yet, try each tx to ensure it's synced locally first
        for tx in tx_srcs.keys():
            Connector(tx).sync_file("logs/agent/rollout.jsonl")
            if rollout.exists():
                rollout.rename(folder / "rollout.jsonl")
                break


def push_rollout_to_trainer(folder: Path):
    """Send the rollout file to TrainAgent."""
    target = folder / "rollout.jsonl"
    if target.exists():
        Connector("TrainAgent").sync_file(str(target), is_pull=False)


def train_once(conn: Connector, control_config: str, traces: Iterable[Path], maybe_load: Path | None = None):
    args = {"control_config": control_config, "trace_path": " ".join(str(p) for p in traces)}
    if maybe_load is not None:
        args["load_path"] = str(maybe_load)
    conn.batch("TrainAgent", "model_train", args)
    apply_until_done(conn)


def pull_trainer_artifacts(exp_name: str, iteration: int, out_folder: Path):
    """Pull model and train.log from TrainAgent; keep a copy in trial folder."""
    # model
    Connector("TrainAgent").sync_file(f"net_util/net_cp/{exp_name}/{iteration}.pt", is_pull=True)
    # train log with timeout
    Connector("TrainAgent").sync_file("net_util/logs/train.log", is_pull=True, timeout=1)

    # stash in trial folder
    tl = Path("net_util/logs/train.log")
    if tl.exists():
        tl.rename(out_folder / "train.log")


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
):
    start_t = time.time()

    # 1) start agents
    start_agents(conn, tx_srcs, eval_mode=args.evaluate)

    # 2) schedule receives & sends
    schedule_all_receives(conn, tx_flows, inter_flows, duration, render=args.render)
    run_tx_and_interference(conn, tx_srcs, inter_srcs, duration)

    # 3) run remote work
    res = apply_until_done(conn)
    res = [r for r in res if r]  # drop {}
    exp_t = time.time()

    # 4) collect artifacts
    folder = now_trial_folder(exp_name)
    pull_rtt_logs(folder, tx_flows)
    move_rollout_from_tx(folder, tx_srcs)
    
    # 5) training (only in train mode)
    if not args.evaluate:
        push_rollout_to_trainer(folder)
        sync_t = time.time()
        
        traces.append(folder / "rollout.jsonl")
        while len(traces) > max_traces:
            traces.pop(0)

        # pick any control_config (all tx share the same key names)
        any_tx = next(iter(tx_srcs))
        control_cfg = tx_srcs[any_tx]["control_config"]

        load_path = Path(f"net_util/net_cp/{exp_name}/{iteration}.pt") if iteration > 0 else None
        train_once(conn, control_cfg, traces, load_path)
        train_t = time.time()

        # 6) pull new model & logs
        next_iter = iteration + 1
        pull_trainer_artifacts(exp_name, next_iter, folder)
        end_t = time.time()

        print(f"Iteration {iteration}:")
        print(f"  Exp time   : {exp_t - start_t:.3f}s")
        print(f"  Sync time  : {sync_t - exp_t:.3f}s")
        print(f"  Train time : {train_t - sync_t:.3f}s")
        print(f"  Pull time  : {end_t - train_t:.3f}s")
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
        return (iteration // args.per_exp_trials) % len(paths)
        
    traces: List[Path] = []
    max_traces = 5
    iteration = 117
    
    path = paths[ path_id(iteration) ]
    print(f"Train config {path}")
    tx_srcs, tx_flows, inter_srcs, inter_flows = create_transmission_config( path, exp_name, conn, is_update=True)
    
    if iteration == 0:
        clean_first_iteration(exp_name)

    while True:
        if iteration % args.per_exp_trials == 0:
            path = paths[ path_id(iteration) ]
            print(f"Train config {path}")
            tx_srcs, tx_flows, inter_srcs, inter_flows = create_transmission_config( path, exp_name, conn, is_update=True)
            
        iteration = run_iteration(
            args, conn, tx_srcs, tx_flows, inter_srcs, inter_flows,
            duration, exp_name, iteration, traces, max_traces
        )


def evaluate(
    args: argparse.Namespace,
    conn: Connector,
    paths: List,
    duration: int,
    exp_name: str,
):
    # single evaluation pass reusing the same building blocks
    for path in paths: 
        print(f"Evaluate config {path}")
        tx_srcs, tx_flows, inter_srcs, inter_flows = create_transmission_config( path, exp_name, conn, is_update=True)
        _ = run_iteration(
            args, conn, tx_srcs, tx_flows, inter_srcs, inter_flows,
            duration, exp_name, iteration=0, traces=[], max_traces=1
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
