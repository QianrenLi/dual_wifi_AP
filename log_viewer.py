#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Log Viewer for PPO training
- Standard UI (Streamlit) + Plotly line charts (zoom preserved via uirevision)
- Watches ALL trial_* under --root (chronological), updates live
- Reward computed via util.trace_collec.trace_filter + flatten_leaves and reward_cfg from control_config
- Training plot uses a global epoch counter (dedup by (trial, local_epoch))
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import streamlit as st
import plotly.graph_objects as go

# --- required project utilities (must be importable) ---
try:
    from util.trace_collec import trace_filter, flatten_leaves  # type: ignore
except Exception as e:
    st.stop()  # If running in Streamlit, stop with message
    raise

# ---------------- CLI (parsed once at startup) ----------------
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Streamlit Log Viewer for PPO")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing trial_* subfolders")
    parser.add_argument("--control-config", type=str, required=True, help="Path to control_config JSON (reads reward_cfg)")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval (seconds)")
    # Note: legacy flag kept for compatibility; plotting now uses sidebar 'Reward view'
    parser.add_argument("--reward-agg", type=str, default="sum", choices=["sum", "mean"], help="Aggregate flattened reward leaves")
    return parser.parse_known_args()

# ---------------- parsing helpers ----------------
_EPOCH_RE = re.compile(r"\[epoch\s+(\d+)\s*/\s*(\d+)\]")
_KV_RE    = re.compile(r"([a-zA-Z_]+)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

def parse_rollout_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

def parse_train_line(line: str) -> Optional[Tuple[int, Dict[str, float]]]:
    m = _EPOCH_RE.search(line)
    if not m:
        return None
    epoch = int(m.group(1))
    metrics: Dict[str, float] = {}
    for k, v, *_ in _KV_RE.findall(line):
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    return epoch, metrics

def flatten_numeric(prefix: str, obj: Any, out: Dict[str, float]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_numeric(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten_numeric(f"{prefix}.{i}" if prefix else str(i), v, out)
    else:
        try:
            out[prefix] = float(obj)
        except Exception:
            pass

# ---------------- file tailing with per-file positions in session ----------------
class FileTailer:
    def __init__(self, path: Path, key_prefix: str):
        self.path = path
        self.key_pos = f"{key_prefix}::pos::{str(path)}"
        self.key_ino = f"{key_prefix}::ino::{str(path)}"
        # initialize state if absent
        st.session_state.setdefault(self.key_pos, 0)
        st.session_state.setdefault(self.key_ino, None)

    def lines(self) -> List[str]:
        out: List[str] = []
        if not self.path.exists():
            return out
        with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
            try:
                st_current_ino = os.fstat(fh.fileno()).st_ino
            except Exception:
                st_current_ino = None

            last_ino = st.session_state[self.key_ino]
            last_pos = st.session_state[self.key_pos]

            # handle rotation (inode changed or file shrank)
            if last_ino is None or last_ino != st_current_ino or last_pos > os.path.getsize(self.path):
                last_pos = 0
                st.session_state[self.key_ino] = st_current_ino

            fh.seek(last_pos)
            chunk = fh.read()
            st.session_state[self.key_pos] = fh.tell()

        if chunk:
            out.extend(chunk.splitlines())
        return out

# ---------------- trial watcher ----------------
def list_trials(root: Path) -> List[Path]:
    trials = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("trial_")]
    return sorted(trials, key=lambda p: p.name)

class MultiTrialWatcher:
    def __init__(self, root: Path):
        self.root = root
        self.trials: List[Path] = []
        self.rollout_tailers: Dict[Path, FileTailer] = {}
        self.train_tailers: Dict[Path, FileTailer] = {}
        # dedup: (trial_path, local_epoch)
        self.seen_train_epochs_key = "seen_train_epochs"
        st.session_state.setdefault(self.seen_train_epochs_key, set())
        self.refresh()

    @property
    def seen_train_epochs(self) -> Set[Tuple[str, int]]:
        # store tuple of (str path, epoch) since Streamlit doesn't like Path in session sometimes
        return st.session_state[self.seen_train_epochs_key]

    def refresh(self):
        current = list_trials(self.root)
        for t in current:
            if t not in self.trials:
                self.trials.append(t)
                self.rollout_tailers[t] = FileTailer(t / "rollout.jsonl", key_prefix="rollout")
                self.train_tailers[t]   = FileTailer(t / "train.log",    key_prefix="train")

    def read_new_rollout(self) -> List[Dict[str, Any]]:
        self.refresh()
        out: List[Dict[str, Any]] = []
        for t in self.trials:
            for line in self.rollout_tailers[t].lines():
                rec = parse_rollout_line(line)
                if rec:
                    out.append(rec)
        return out

    def read_new_train(self) -> List[Tuple[Path, int, Dict[str, float]]]:
        self.refresh()
        results: List[Tuple[Path, int, Dict[str, float]]] = []
        for t in self.trials:
            for line in self.train_tailers[t].lines():
                parsed = parse_train_line(line)
                if not parsed:
                    continue
                epoch, metrics = parsed
                key = (str(t), epoch)
                if key in self.seen_train_epochs:
                    continue
                self.seen_train_epochs.add(key)
                results.append((t, epoch, metrics))
        return results

# ---------------- reward from trace_filter + flatten_leaves ----------------
def build_reward_from_record(record: Dict[str, Any], reward_descriptor: Optional[Dict[str, Any]], agg: str) -> float:
    """Return per-slot reward as sum of flattened leaves (default behavior)."""
    if reward_descriptor is None:
        return 0.0
    filtered = trace_filter(record, reward_descriptor)
    leaves = flatten_leaves(filtered)
    return float(sum(leaves))

# ---------------- helpers for plotting reward modes ----------------
def discounted_running_sum(xs: List[float], gamma: float) -> List[float]:
    acc = 0.0
    out: List[float] = []
    for r in xs:
        acc = acc * gamma + r
        out.append(acc)
    return out

# ---------------- data model in session ----------------
def init_session_data():
    st.session_state.setdefault("global_step", 0)
    st.session_state.setdefault("global_epoch", 0)
    st.session_state.setdefault("t_rollout", [])
    st.session_state.setdefault("reward_values", [])  # kept for compatibility
    st.session_state.setdefault("reward_raw", [])     # element-wise per-slot reward
    st.session_state.setdefault("t_train", [])
    st.session_state.setdefault("rollout_series", {})  # key -> list
    st.session_state.setdefault("train_series", {})    # key -> list
    st.session_state.setdefault("rollout_keys_all", [])  # discovered numeric keys (stats.* or res.*)
    st.session_state.setdefault("train_keys_all", [])
    # preserve zoom with Plotly: uirevision tokens
    st.session_state.setdefault("rollout_uirev", "rollout-0")
    st.session_state.setdefault("train_uirev", "train-0")

def append_rollout(rec: Dict[str, Any], reward_cfg: Optional[Dict[str, Any]], agg: str):
    st.session_state["global_step"] += 1
    st.session_state["t_rollout"].append(st.session_state["global_step"])
    r = build_reward_from_record(rec, reward_cfg, agg)  # element-wise reward at this step
    st.session_state["reward_raw"].append(float(r))
    st.session_state["reward_values"].append(float(r))  # kept for back-compat

    # flatten numeric leaves
    flat: Dict[str, float] = {}
    flatten_numeric("", rec, flat)
    for k, v in flat.items():
        if k.startswith("stats.") or k.startswith("res."):
            if k not in st.session_state["rollout_keys_all"]:
                st.session_state["rollout_keys_all"].append(k)
        st.session_state["rollout_series"].setdefault(k, []).append(v)

def append_train(metrics: Dict[str, float]):
    st.session_state["global_epoch"] += 1
    st.session_state["t_train"].append(st.session_state["global_epoch"])
    for k, v in metrics.items():
        if k not in st.session_state["train_keys_all"]:
            st.session_state["train_keys_all"].append(k)
        st.session_state["train_series"].setdefault(k, []).append(v)

# ---------------- plotting ----------------
def plot_rollout(selected_keys: List[str], uirev: str, reward_mode: str, gamma: float):
    t = st.session_state["t_rollout"]
    if not t:
        return go.Figure()
    fig = go.Figure()
    # reward first
    if "reward" in selected_keys:
        raw = st.session_state["reward_raw"]
        ys = raw if reward_mode == "element" else discounted_running_sum(raw, gamma)
        xs = t[:len(ys)]
        if xs and ys:
            name = "reward (element)" if reward_mode == "element" else f"reward (acc, γ={gamma:.3f})"
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))
    # other selected keys
    for k in selected_keys:
        if k == "reward":
            continue
        ys = st.session_state["rollout_series"].get(k, [])
        xs = t[:len(ys)]
        if xs and ys:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=k, hovertemplate="x=%{x}<br>y=%{y}<extra>"+k+"</extra>"))
    fig.update_layout(
        xaxis_title="global step",
        yaxis_title="value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        uirevision=uirev,  # preserves zoom/pan across updates
    )
    return fig

def plot_train(selected_keys: List[str], uirev: str):
    t = st.session_state["t_train"]
    if not t:
        return go.Figure()
    fig = go.Figure()
    for k in selected_keys:
        ys = st.session_state["train_series"].get(k, [])
        xs = t[:len(ys)]
        if xs and ys:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=k, hovertemplate="epoch=%{x}<br>y=%{y}<extra>"+k+"</extra>"))
    fig.update_layout(
        xaxis_title="global epoch",
        yaxis_title="value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        uirevision=uirev,
    )
    return fig

# ---------------- Streamlit app ----------------
def main():
    args, _ = parse_cli_args()
    root = Path(args.root).resolve()
    if not root.exists():
        st.error(f"Root not found: {root}")
        st.stop()
    control_config_path = Path(args.control_config).resolve()
    if not control_config_path.exists():
        st.error(f"control_config not found: {control_config_path}")
        st.stop()

    # load control_config
    try:
        with open(control_config_path, "r", encoding="utf-8") as f:
            cc = json.load(f)
        reward_cfg = cc.get("reward_cfg", None)
    except Exception as e:
        st.error(f"Failed to load control_config: {e}")
        st.stop()

    # page config
    st.set_page_config(page_title="PPO Log Viewer", layout="wide")

    # topbar
    st.title("PPO Log Viewer")
    st.caption(f"Root: `{root}` · control_config: `{control_config_path}`")

    # sidebar controls
    with st.sidebar:
        st.header("Controls")
        refresh = st.number_input("Refresh interval (sec)", min_value=0.2, value=float(args.refresh), step=0.2, format="%.1f")

        # Reward view (plotting only): element-wise vs accumulated discounted
        reward_mode = st.selectbox("Reward view", options=["element", "acc"], index=0)
        gamma = 0.99
        if reward_mode == "acc":
            gamma = st.number_input("Discount γ", min_value=0.0, max_value=1.0, value=0.99, step=0.01, format="%.4f")

        st.divider()
        # filtering + selection
        r_filter = st.text_input("Filter rollout keys (substring)", value="")
        t_filter = st.text_input("Filter train keys (substring)", value="")
        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        st.caption("Zoom and pan are preserved between refreshes.")

    init_session_data()
    watcher = MultiTrialWatcher(root)

    # ingest new lines
    for rec in watcher.read_new_rollout():
        # pass args.reward_agg only to keep signature; plotting ignores it now
        append_rollout(rec, reward_cfg, args.reward_agg)
    for _trial, _epoch, metrics in watcher.read_new_train():
        append_train(metrics)

    # build key lists after ingest
    rollout_keys_all = ["reward"] + sorted([k for k in st.session_state["rollout_keys_all"] if k.startswith("stats.") or k.startswith("res.")])
    train_keys_all = sorted(st.session_state["train_keys_all"])

    # filter
    if r_filter:
        rollout_keys_view = ["reward"] + [k for k in rollout_keys_all if k != "reward" and r_filter.lower() in k.lower()]
    else:
        rollout_keys_view = rollout_keys_all

    if t_filter:
        train_keys_view = [k for k in train_keys_all if t_filter.lower() in k.lower()]
    else:
        train_keys_view = train_keys_all

    # selection widgets (standard UI)
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Rollout")
        default_rollout = ["reward"]  # always include reward by default
        sel_rollout = st.multiselect(
            "Select rollout series",
            options=rollout_keys_view,
            default=[k for k in default_rollout if k in rollout_keys_view],
            key="sel_rollout_multiselect",
        )
        fig_rollout = plot_rollout(sel_rollout, uirev=st.session_state["rollout_uirev"],
                                   reward_mode=reward_mode, gamma=gamma)
        st.plotly_chart(fig_rollout, use_container_width=True)
    with col_right:
        st.subheader("Training")
        default_train = ["loss", "pol_loss", "val_loss"]
        defaults = [k for k in default_train if k in train_keys_view]
        sel_train = st.multiselect(
            "Select training series",
            options=train_keys_view,
            default=defaults,
            key="sel_train_multiselect",
        )
        fig_train = plot_train(sel_train, uirev=st.session_state["train_uirev"])
        st.plotly_chart(fig_train, use_container_width=True)

    # auto refresh
    if auto_refresh:
        time.sleep(refresh)
        st.rerun()

if __name__ == "__main__":
    main()
