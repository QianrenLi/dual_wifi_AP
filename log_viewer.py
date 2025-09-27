#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Log Viewer for PPO training (refactored, stable, manual refresh)

Goals
- Multiple independent windows for both rollout metrics and training metrics
- Smooth plotting (ScatterGL, optional downsampling when points are huge)
- Readable structure, fewer side effects, safer state management
- No automatic refresh by default (manual Refresh button)
- Stable UI: no sleep/rerun loops; preserve zoom via Plotly uirevision

Usage
streamlit run ppo_log_viewer_refactored.py \
  -- --root /path/to/root --control-config /path/to/control.json --reward-agg sum --recent 3

Notes
- Trials are folders named trial_YYYYMMDD-HHMMSS (preferred) or any trial_* (mtime fallback)
- We deduplicate training epochs across trials via (trial_path, epoch)
- When switching "Recent N", we reset cursors and ingest full history once
- Reward view supports element-wise or discounted accumulated (γ)

"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import streamlit as st
import plotly.graph_objects as go

# --- required project utilities (must be importable) ---
try:
    from util.trace_collec import trace_filter, flatten_leaves  # type: ignore
except Exception as e:
    st.error("Failed to import util.trace_collec: " + str(e))
    st.stop()

# ==========================================================
# CLI
# ==========================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamlit Log Viewer for PPO (refactored)")
    parser.add_argument("--root", type=str, required=True, help="Directory containing trial_* folders")
    parser.add_argument("--control-config", type=str, required=True, help="JSON with reward_cfg")
    parser.add_argument("--reward-agg", type=str, default="sum", choices=["sum", "mean"], help="Aggregate for reward leaves")
    parser.add_argument("--recent", type=int, default=0, help="Only read the most recent N trial folders; 0 = all")
    return parser.parse_args()

# ==========================================================
# Parsing helpers
# ==========================================================

_EPOCH_RE = re.compile(r"\[epoch\s+(\d+)\s*/\s*(\d+)\]")
_KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_\./]*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_TS_RE = re.compile(r"^trial_(\d{8}-\d{6})$")  # trial_YYYYMMDD-HHMMSS


def parse_rollout_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_train_line(line: str) -> Optional[Tuple[int, Dict[str, float]]]:
    m = _EPOCH_RE.search(line)
    if not m:
        return None
    epoch = int(m.group(1))
    metrics: Dict[str, float] = {}
    for k, v in _KV_RE.findall(line):
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    return epoch, metrics


def flatten_numeric(prefix: str, obj: Any, out: Dict[str, float]) -> None:
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

# ==========================================================
# Trial discovery & sort
# ==========================================================


def _trial_sort_key(p: Path) -> Tuple[int, str]:
    """Prefer timestamp in folder name; fallback to mtime."""
    m = _TS_RE.match(p.name)
    if m:
        ts = m.group(1)
        try:
            dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
            return (int(dt.timestamp()), p.name)
        except Exception:
            pass
    try:
        return (int(p.stat().st_mtime), p.name)
    except Exception:
        return (0, p.name)


@st.cache_data(show_spinner=False)
def list_trials_cached(root: str) -> List[str]:
    root_path = Path(root)
    trials = [str(p) for p in root_path.iterdir() if p.is_dir() and p.name.startswith("trial_")]
    trials_sorted = sorted(trials, key=lambda s: _trial_sort_key(Path(s)), reverse=True)
    return trials_sorted


def select_recent(trials: List[str], recent_n: int) -> List[str]:
    if recent_n and recent_n > 0:
        return trials[: recent_n]
    return trials

# ==========================================================
# File tailing (manual, cursor in session_state)
# ==========================================================

@dataclass
class TailState:
    pos: int = 0
    ino: Optional[int] = None


class FileTailer:
    def __init__(self, path: Path, ns_key: str):
        self.path = path
        self.ns_key = ns_key
        if self.ns_key not in st.session_state:
            st.session_state[self.ns_key] = TailState()

    @property
    def state(self) -> TailState:
        return st.session_state[self.ns_key]

    def reset(self) -> None:
        st.session_state[self.ns_key] = TailState()

    def read_new_lines(self) -> List[str]:
        out: List[str] = []
        if not self.path.exists():
            return out
        with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
            try:
                current_ino = os.fstat(fh.fileno()).st_ino
            except Exception:
                current_ino = None
            stt: TailState = self.state
            size_now = os.path.getsize(self.path)
            # rotation/truncation
            if stt.ino is None or stt.ino != current_ino or stt.pos > size_now:
                stt.pos = 0
                stt.ino = current_ino
            fh.seek(stt.pos)
            chunk = fh.read()
            stt.pos = fh.tell()
        if chunk:
            out.extend(chunk.splitlines())
        return out

# ==========================================================
# Watcher across multiple trials
# ==========================================================

class MultiTrialWatcher:
    def __init__(self, root: Path, recent_n: int):
        self.root = root
        self.recent_n = recent_n
        self.trials: List[Path] = []
        self.rollout_tailers: Dict[str, FileTailer] = {}
        self.train_tailers: Dict[str, FileTailer] = {}
        if "seen_train_epochs" not in st.session_state:
            st.session_state["seen_train_epochs"] = set()
        self.rebuild()

    def set_recent(self, n: int) -> None:
        if n != self.recent_n:
            self.recent_n = n
            self.rebuild(reset_positions=True)

    def rebuild(self, reset_positions: bool = False) -> None:
        all_trials = list_trials_cached(str(self.root))
        chosen = select_recent(all_trials, self.recent_n)
        self.trials = [Path(p) for p in chosen]
        # Create tailers for current trials only
        new_rollout: Dict[str, FileTailer] = {}
        new_train: Dict[str, FileTailer] = {}
        for t in self.trials:
            rkey = f"rollout::{t}"
            tkey = f"train::{t}"
            rt = FileTailer(t / "rollout.jsonl", rkey)
            tt = FileTailer(t / "train.log", tkey)
            if reset_positions:
                rt.reset()
                tt.reset()
            new_rollout[str(t)] = rt
            new_train[str(t)] = tt
        self.rollout_tailers = new_rollout
        self.train_tailers = new_train
        if reset_positions:
            st.session_state["seen_train_epochs"] = set()

    def ingest_rollout(self) -> List[Dict[str, Any]]:
        # oldest -> newest across the selected trials for temporal continuity
        out: List[Dict[str, Any]] = []
        for t in reversed(self.trials):
            for line in self.rollout_tailers[str(t)].read_new_lines():
                rec = parse_rollout_line(line)
                if rec is not None:
                    out.append(rec)
        return out

    def ingest_train(self) -> List[Tuple[Path, int, Dict[str, float]]]:
        results: List[Tuple[Path, int, Dict[str, float]]] = []
        seen = st.session_state["seen_train_epochs"]
        for t in reversed(self.trials):
            for line in self.train_tailers[str(t)].read_new_lines():
                parsed = parse_train_line(line)
                if not parsed:
                    continue
                epoch, metrics = parsed
                key = (str(t), epoch)
                if key in seen:
                    continue
                seen.add(key)
                results.append((t, epoch, metrics))
        return results

# ==========================================================
# Reward & series helpers
# ==========================================================


def build_reward_from_record(record: Dict[str, Any], reward_descriptor: Optional[Dict[str, Any]], agg: str) -> float:
    if reward_descriptor is None:
        return 0.0
    filtered = trace_filter(record, reward_descriptor)
    leaves = flatten_leaves(filtered)
    if not leaves:
        return 0.0
    if agg == "mean":
        return float(sum(leaves) / max(1, len(leaves)))
    return float(sum(leaves))


def discounted_running_sum(xs: List[float], gamma: float) -> List[float]:
    acc = 0.0
    out: List[float] = []
    for r in xs:
        acc = acc * gamma + r
        out.append(acc)
    return out

# ==========================================================
# Session state schema
# ==========================================================


def init_session(args: argparse.Namespace) -> None:
    st.session_state.setdefault("recent_n", int(args.recent))
    st.session_state.setdefault("global_step", 0)
    st.session_state.setdefault("global_epoch", 0)
    st.session_state.setdefault("t_rollout", [])
    st.session_state.setdefault("reward_raw", [])
    st.session_state.setdefault("t_train", [])
    st.session_state.setdefault("rollout_series", {})  # key -> list[float]
    st.session_state.setdefault("train_series", {})    # key -> list[float]
    st.session_state.setdefault("rollout_keys_all", [])
    st.session_state.setdefault("train_keys_all", [])
    st.session_state.setdefault("rollout_windows", [])  # list[{id, selected_keys}]
    st.session_state.setdefault("train_windows", [])    # list[{id, selected_keys}]
    st.session_state.setdefault("win_counter", 0)
    st.session_state.setdefault("reward_mode", "element")
    st.session_state.setdefault("gamma", 0.99)


def reset_timeseries() -> None:
    st.session_state["global_step"] = 0
    st.session_state["global_epoch"] = 0
    st.session_state["t_rollout"] = []
    st.session_state["reward_raw"] = []
    st.session_state["t_train"] = []
    st.session_state["rollout_series"] = {}
    st.session_state["train_series"] = {}
    st.session_state["rollout_keys_all"] = []
    st.session_state["train_keys_all"] = []


# ==========================================================
# Append helpers
# ==========================================================


def append_rollout(rec: Dict[str, Any], reward_cfg: Optional[Dict[str, Any]], agg: str) -> None:
    st.session_state["global_step"] += 1
    st.session_state["t_rollout"].append(st.session_state["global_step"])
    r = build_reward_from_record(rec, reward_cfg, agg)
    st.session_state["reward_raw"].append(float(r))
    flat: Dict[str, float] = {}
    flatten_numeric("", rec, flat)
    for k, v in flat.items():
        if k.startswith("stats.") or k.startswith("res."):
            if k not in st.session_state["rollout_keys_all"]:
                st.session_state["rollout_keys_all"].append(k)
        st.session_state["rollout_series"].setdefault(k, []).append(v)


def append_train(metrics: Dict[str, float]) -> None:
    st.session_state["global_epoch"] += 1
    st.session_state["t_train"].append(st.session_state["global_epoch"])
    for k, v in metrics.items():
        if k not in st.session_state["train_keys_all"]:
            st.session_state["train_keys_all"].append(k)
        st.session_state["train_series"].setdefault(k, []).append(v)

# ==========================================================
# Plotting (ScatterGL + optional downsampling)
# ==========================================================


def _maybe_downsample(xs: List[float], ys: List[float], max_points: int = 6000) -> Tuple[List[float], List[float]]:
    n = min(len(xs), len(ys))
    if n <= max_points:
        return xs[:n], ys[:n]
    # simple stride-based downsampling (fast, preserves general shape)
    stride = max(1, n // max_points)
    xs_ds = xs[::stride]
    ys_ds = ys[::stride]
    # ensure last point is included
    if xs_ds[-1] != xs[n - 1]:
        xs_ds.append(xs[n - 1])
        ys_ds.append(ys[n - 1])
    return xs_ds, ys_ds


def plot_rollout(selected_keys: List[str], reward_mode: str, gamma: float, uirev: str) -> go.Figure:
    t = st.session_state["t_rollout"]
    fig = go.Figure()
    if not t:
        fig.update_layout(uirevision=uirev)
        return fig

    # Reward first (if selected)
    if "reward" in selected_keys:
        raw = st.session_state["reward_raw"]
        ys = raw if reward_mode == "element" else discounted_running_sum(raw, gamma)
        xs = t[: len(ys)]
        xs, ys = _maybe_downsample(xs, ys)
        if xs and ys:
            name = "reward (element)" if reward_mode == "element" else f"reward (acc, γ={gamma:.3f})"
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=name))

    # Other rollout series
    for k in selected_keys:
        if k == "reward":
            continue
        ys = st.session_state["rollout_series"].get(k, [])
        xs = t[: len(ys)]
        xs, ys = _maybe_downsample(xs, ys)
        if xs and ys:
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=k))

    fig.update_layout(
        uirevision=uirev,
        xaxis_title="global step",
        yaxis_title="value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_train(selected_keys: List[str], uirev: str) -> go.Figure:
    t = st.session_state["t_train"]
    fig = go.Figure()
    if not t:
        fig.update_layout(uirevision=uirev)
        return fig
    for k in selected_keys:
        ys = st.session_state["train_series"].get(k, [])
        xs = t[: len(ys)]
        xs, ys = _maybe_downsample(xs, ys)
        if xs and ys:
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=k))
    fig.update_layout(
        uirevision=uirev,
        xaxis_title="global epoch",
        yaxis_title="value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig

# ==========================================================
# Main app
# ==========================================================


def main() -> None:
    args = parse_cli_args()
    root = Path(args.root).resolve()
    if not root.exists():
        st.error(f"Root not found: {root}")
        st.stop()
    control_path = Path(args.control_config).resolve()
    if not control_path.exists():
        st.error(f"control_config not found: {control_path}")
        st.stop()

    st.set_page_config(page_title="PPO Log Viewer (refactored)", layout="wide")
    st.title("PPO Log Viewer")
    st.caption(f"Root: `{root}` · control_config: `{control_path}` · Mode: Manual refresh")

    # Load control_config (for reward_cfg)
    try:
        with open(control_path, "r", encoding="utf-8") as f:
            reward_cfg = json.load(f).get("reward_cfg", None)
    except Exception as e:
        st.error(f"Failed to load control_config: {e}")
        st.stop()

    # Init state
    init_session(args)

    # Watcher singleton in state
    if "watcher" not in st.session_state:
        st.session_state["watcher"] = MultiTrialWatcher(root, st.session_state["recent_n"])  # type: ignore
    watcher: MultiTrialWatcher = st.session_state["watcher"]

    # --- Initial one-time ingest so charts show without pressing buttons ---
    if "bootstrapped" not in st.session_state:
        st.session_state["bootstrapped"] = False
    if not st.session_state["bootstrapped"]:
        for rec in watcher.ingest_rollout():
            append_rollout(rec, reward_cfg, args.reward_agg)
        for _, _, metrics in watcher.ingest_train():
            append_train(metrics)
        st.session_state["bootstrapped"] = True


    # Sidebar controls (no auto-refresh; explicit buttons only)
    with st.sidebar:
        st.header("Controls")
        # Change recent N
        recent_input = st.number_input(
            "Recent trials (0 = all)",
            min_value=0, step=1, value=int(st.session_state["recent_n"]),
            help="Limit to N most recent trial_* folders by timestamp.",
        )
        colA, colB = st.columns(2)
        with colA:
            apply_recent = st.button("Apply recent N")
        with colB:
            refresh_trials = st.button("Reload trial list")

        st.divider()
        st.session_state["reward_mode"] = st.selectbox(
            "Reward view",
            ["element", "acc"],
            index=0 if st.session_state["reward_mode"] == "element" else 1,
        )
        if st.session_state["reward_mode"] == "acc":
            st.session_state["gamma"] = st.number_input(
                "Discount γ",
                min_value=0.0, max_value=1.0,
                value=float(st.session_state["gamma"]), step=0.01, format="%.4f",
            )

        st.divider()
        add_roll = st.button("➕ Add rollout window")
        add_train = st.button("➕ Add training window")
        st.caption("Windows are independent; zoom/pan persist until you change series.")

        st.divider()
        ingest_now = st.button("🔄 Refresh now (ingest new lines)")
        full_reload = st.button("⟲ Full reload (reset cursors & re-read)")

    # Handle sidebar actions
    if refresh_trials:
        list_trials_cached.clear()  # invalidate cache
    if apply_recent:
        st.session_state["recent_n"] = int(recent_input)
        reset_timeseries()
        watcher.set_recent(st.session_state["recent_n"])  # resets cursors too
        # initial ingest of full history after reset
        for rec in watcher.ingest_rollout():
            append_rollout(rec, reward_cfg, args.reward_agg)
        for _, _, metrics in watcher.ingest_train():
            append_train(metrics)
    if add_roll:
        st.session_state["win_counter"] += 1
        st.session_state["rollout_windows"].append({"id": st.session_state["win_counter"], "selected_keys": ["reward"]})
    if add_train:
        st.session_state["win_counter"] += 1
        st.session_state["train_windows"].append({"id": st.session_state["win_counter"], "selected_keys": []})
    if full_reload:
        reset_timeseries()
        watcher.rebuild(reset_positions=True)
        for rec in watcher.ingest_rollout():
            append_rollout(rec, reward_cfg, args.reward_agg)
        for _, _, metrics in watcher.ingest_train():
            append_train(metrics)
    if ingest_now:
        for rec in watcher.ingest_rollout():
            append_rollout(rec, reward_cfg, args.reward_agg)
        for _, _, metrics in watcher.ingest_train():
            append_train(metrics)

    # Show active trials
    with st.expander("Active trials", expanded=False):
        all_trials = list_trials_cached(str(root))
        chosen = select_recent(all_trials, st.session_state["recent_n"])  # newest first
        if not chosen:
            st.warning("No trial_* folders found under root.")
        else:
            st.write(f"Reading {len(chosen)} most recent trial(s):")
            for p in chosen:
                st.write("- " + Path(p).name)

    # Live counters so you can see something loaded
    st.success(
        f"Loaded rollout points: {len(st.session_state['t_rollout'])} · "
        f"train epochs: {len(st.session_state['t_train'])}"
    )

    # Available keys
    rollout_keys_all = ["reward"] + sorted([k for k in st.session_state["rollout_keys_all"] if k.startswith("stats.") or k.startswith("res.")])
    train_keys_all = sorted(st.session_state["train_keys_all"]) or [
        # good defaults if present
        k for k in ["loss", "pol_loss", "val_loss", "entropy", "kl", "clipfrac"] if k in st.session_state["train_series"]
    ]

    # Layout: left rollout windows, right training windows
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rollout Windows")
        if not st.session_state["rollout_windows"]:
            st.info("Use the sidebar to add a rollout window (e.g., reward + stats.link.192.168.3.25.tx_mbit_s).")
        remain_roll = []
        for w in st.session_state["rollout_windows"]:
            wid = w["id"]
            with st.container(border=True):
                # series selector
                default_sel = [k for k in w.get("selected_keys", []) if k in rollout_keys_all] or ["reward"]
                sel = st.multiselect(
                    f"Series (rollout win {wid})",
                    options=rollout_keys_all,
                    default=default_sel,
                    key=f"sel_roll_{wid}",
                )
                w["selected_keys"] = sel
                # actions & chart
                c1, c2 = st.columns([1, 1])
                with c1:
                    rm = st.button("✖ Remove", key=f"del_roll_{wid}")
                with c2:
                    clr = st.button("🧹 Clear series data", key=f"clr_roll_{wid}")
                if clr:
                    # Clear only plotted series for this window
                    for k in sel:
                        if k == "reward":
                            continue
                        st.session_state["rollout_series"].pop(k, None)
                fig = plot_rollout(sel, st.session_state["reward_mode"], st.session_state["gamma"], uirev=f"rollout-{wid}")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_roll_{wid}")
                if not rm:
                    remain_roll.append(w)
        st.session_state["rollout_windows"] = remain_roll

    with col_right:
        st.subheader("Training Windows")
        if not st.session_state["train_windows"]:
            st.info("Use the sidebar to add a training window (e.g., loss / pol_loss / val_loss).")
        remain_train = []
        for w in st.session_state["train_windows"]:
            wid = w["id"]
            with st.container(border=True):
                default_sel = [k for k in w.get("selected_keys", []) if k in train_keys_all]
                sel = st.multiselect(
                    f"Series (train win {wid})",
                    options=train_keys_all or sorted(st.session_state["train_series"].keys()),
                    default=default_sel,
                    key=f"sel_train_{wid}",
                )
                w["selected_keys"] = sel
                c1, c2 = st.columns([1, 1])
                with c1:
                    rm = st.button("✖ Remove", key=f"del_train_{wid}")
                with c2:
                    clr = st.button("🧹 Clear series data", key=f"clr_train_{wid}")
                if clr:
                    for k in sel:
                        st.session_state["train_series"].pop(k, None)
                fig = plot_train(sel, uirev=f"train-{wid}")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_train_{wid}")
                if not rm:
                    remain_train.append(w)
        st.session_state["train_windows"] = remain_train


if __name__ == "__main__":
    main()
