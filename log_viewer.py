#!/usr/bin/env python3
"""
Live PPO Log Viewer (multi-trial, reward via trace_filter)
- Watches ALL trial_* folders under --root and concatenates data chronologically.
- Reward is computed like your trainer: trace_filter(record, reward_cfg) -> flatten_leaves -> aggregate.
- Training plot is global across trials (per-trial epochs get an offset and duplicates are suppressed).
- Dynamic metric toggles (rollout/train) via on-plot CheckButtons.

Examples
--------
python log_viewer.py --root exp_trace/system_verify --control-config net_util/net_config/STA1_STA2.json
python log_viewer.py --root exp_trace/system_verify --control-config ... --refresh 0.5 \
    --rollout-keys stats.flow_stat.*.throughput stats.device_stat.link.*.tx_mbit_s \
    --train-keys loss pol_loss val_loss entropy kl clipfrac lr avg_return exp_var
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# --- your utilities (required) ---
try:
    # must exist in your repo per prior code
    from util.trace_collec import trace_filter, flatten_leaves  # type: ignore
except Exception as e:
    print("[ERR] Could not import util.trace_collec.{trace_filter, flatten_leaves}. "
          "Please ensure your PYTHONPATH includes the project root.", file=sys.stderr)
    raise

# ---------------- parsing helpers ----------------
def parse_rollout_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None

_EPOCH_RE = re.compile(r"\[epoch\s+(\d+)\s*/\s*(\d+)\]")
_KV_RE    = re.compile(r"([a-zA-Z_]+)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

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

# ---------------- tailer & trial watcher ----------------
class FileTailer:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._pos = 0

    def _ensure_open(self):
        if self._fh is None and self.path.exists():
            self._fh = open(self.path, "r", encoding="utf-8", errors="ignore")
            self._fh.seek(self._pos)

    def lines(self) -> List[str]:
        out: List[str] = []
        self._ensure_open()
        if self._fh is None:
            return out
        chunk = self._fh.read()
        if chunk:
            self._pos = self._fh.tell()
            out.extend(chunk.splitlines())
        return out

def list_trials(root: Path) -> List[Path]:
    trials = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("trial_")]
    return sorted(trials, key=lambda p: p.name)  # chronological by name

class MultiTrialWatcher:
    def __init__(self, root: Path):
        self.root = root
        self.trials: List[Path] = []
        self.rollout_tailers: Dict[Path, FileTailer] = {}
        self.train_tailers: Dict[Path, FileTailer] = {}
        self.last_seen_epoch: Dict[Path, int] = {}
        self.refresh_trials()

    def refresh_trials(self):
        current = list_trials(self.root)
        for t in current:
            if t not in self.trials:
                self.trials.append(t)
                self.rollout_tailers[t] = FileTailer(t / "rollout.jsonl")
                self.train_tailers[t]   = FileTailer(t / "train.log")
                self.last_seen_epoch[t] = 0

    def read_new_rollout(self) -> List[Dict[str, Any]]:
        self.refresh_trials()
        out: List[Dict[str, Any]] = []
        for t in self.trials:
            for line in self.rollout_tailers[t].lines():
                rec = parse_rollout_line(line)
                if rec:
                    out.append(rec)
        return out

    def read_new_train(self) -> List[Tuple[int, Dict[str, float]]]:
        """
        Returns (global_epoch, metrics). Global epoch is built by offsetting
        per-trial epochs with the max epoch observed in all earlier trials.
        """
        self.refresh_trials()
        # compute offsets = cumulative max epochs of prior trials
        offsets: Dict[Path, int] = {}
        cum = 0
        for t in self.trials:
            offsets[t] = cum
            cum += self.last_seen_epoch.get(t, 0)

        results: List[Tuple[int, Dict[str, float]]] = []
        for t in self.trials:
            for line in self.train_tailers[t].lines():
                parsed = parse_train_line(line)
                if not parsed:
                    continue
                epoch, metrics = parsed
                if epoch > self.last_seen_epoch[t]:
                    self.last_seen_epoch[t] = epoch
                global_epoch = offsets[t] + epoch
                results.append((global_epoch, metrics))
        return results

# ---------------- reward via trace_filter + flatten_leaves ----------------
def build_reward_from_record(record: Dict[str, Any],
                             reward_descriptor: Optional[Dict[str, Any]],
                             agg: str = "sum") -> float:
    """
    Reconstruct reward per record, matching your training:
      - filtered = trace_filter(record, reward_descriptor)
      - leaves   = flatten_leaves(filtered)
      - reward   = sum(leaves) | mean(leaves)
    """
    if reward_descriptor is None:
        return 0.0
    filtered = trace_filter(record, reward_descriptor)
    leaves = flatten_leaves(filtered)
    if not leaves:
        return 0.0
    if agg == "mean":
        return float(sum(leaves) / len(leaves))
    # default: sum
    return float(sum(leaves))

# ---------------- live plot ----------------
class LivePlotAll:
    def __init__(self,
                 root: Path,
                 control_config_path: Path,
                 reward_agg: str,
                 refresh: float,
                 pre_rollout: List[str],
                 pre_train: List[str]):
        self.root = root
        self.refresh = max(0.2, refresh)
        self.reward_agg = reward_agg

        # read control_config → reward_cfg (used by trace_filter)
        try:
            with open(control_config_path, "r", encoding="utf-8") as f:
                cc = json.load(f)
            self.reward_cfg = cc.get("reward_cfg", None)
        except Exception as e:
            print(f"[ERR] Failed to load control_config: {e}", file=sys.stderr)
            self.reward_cfg = None
            

        self.watcher = MultiTrialWatcher(self.root)

        # global x axes
        self.global_step = 0               # rollout
        self._global_epoch_seen = set()    # backing field for property

        # data buffers
        self.t_rollout: List[int] = []
        self.rollout_series: Dict[str, List[float]] = {}
        self.t_reward: List[int] = []
        self.reward_values: List[float] = []

        self.t_train: List[int] = []
        self.train_series: Dict[str, List[float]] = {}

        # discovered keys
        self.rollout_keys: List[str] = []
        self.train_keys: List[str] = []

        # selections
        self.sel_rollout: Dict[str, bool] = {k: True for k in pre_rollout}
        self.sel_train: Dict[str, bool] = {k: True for k in pre_train}

        # figure
        self.fig = plt.figure(figsize=(12, 7))
        self.ax_rollout = self.fig.add_axes([0.08, 0.55, 0.72, 0.38])
        self.ax_train   = self.fig.add_axes([0.08, 0.08, 0.72, 0.38])

        self.ax_rollout.set_title(f"Rollout metrics — ALL trials in {self.root.name}")
        self.ax_rollout.set_xlabel("global step")
        self.ax_train.set_title("Training metrics — ALL trials")
        self.ax_train.set_xlabel("global epoch")

        self.cb_ax_rollout = self.fig.add_axes([0.82, 0.55, 0.16, 0.38])
        self.cb_ax_train   = self.fig.add_axes([0.82, 0.08, 0.16, 0.38])
        self.cb_rollout = None
        self.cb_train = None

        # timer
        self.timer = self.fig.canvas.new_timer(interval=int(self.refresh * 1000))
        self.timer.add_callback(self._tick)
        self.timer.start()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    @property
    def global_epoch_seen(self):
        # read-only accessor for duplicate suppression
        return self._global_epoch_seen

    def _on_key(self, ev):
        if ev.key == "r":
            self.ax_rollout.relim(); self.ax_rollout.autoscale()
            self.ax_train.relim(); self.ax_train.autoscale()
            self.fig.canvas.draw_idle()

    def _ensure_rollout_cb(self):
        if self.cb_rollout is not None:
            return
        labels = ["reward"] + self.rollout_keys
        actives = [True] + [self.sel_rollout.get(k, False) for k in self.rollout_keys]
        self.cb_ax_rollout.cla()
        self.cb_rollout = CheckButtons(self.cb_ax_rollout, labels=labels, actives=actives)
        def _clk(label):
            self.sel_rollout[label] = not self.sel_rollout.get(label, label == "reward")
        self.cb_rollout.on_clicked(_clk)

    def _ensure_train_cb(self):
        if self.cb_train is not None:
            return
        defaults = ["loss", "pol_loss", "val_loss", "entropy", "kl", "clipfrac", "lr", "avg_return", "exp_var"]
        labels, seen = [], set()
        for k in self.train_keys + defaults:
            if k not in seen:
                labels.append(k); seen.add(k)
        actives = [self.sel_train.get(k, k in ("loss", "pol_loss")) for k in labels]
        self.cb_ax_train.cla()
        self.cb_train = CheckButtons(self.cb_ax_train, labels=labels, actives=actives)
        def _clk(label):
            self.sel_train[label] = not self.sel_train.get(label, False)
        self.cb_train.on_clicked(_clk)

    def _tick(self):
        # ---- rollout across all trials ----
        for rec in self.watcher.read_new_rollout():
            self.global_step += 1
            self.t_rollout.append(self.global_step)

            # reward via trace_filter + flatten_leaves
            rew = build_reward_from_record(rec, self.reward_cfg, self.reward_agg)
            self.t_reward.append(self.global_step)
            self.reward_values.append(float(rew))

            # flatten numeric leaves as plot candidates
            flat: Dict[str, float] = {}
            flatten_numeric("", rec, flat)
            for k, v in flat.items():
                # show primarily useful signals
                if any(x in k for x in (".throughput", ".bitrate", ".tx_mbit_s",
                                        ".signal_dbm", ".app_buff", ".rtt",
                                        "res.value", "res.log_prob", "res.action")):
                    if k not in self.rollout_keys:
                        self.rollout_keys.append(k)
                self.rollout_series.setdefault(k, []).append(v)

        # ---- training across all trials ----
        for gepoch, metrics in self.watcher.read_new_train():
            # ensure strict monotonic epochs & no dupes
            if gepoch in self.global_epoch_seen:
                continue
            self.global_epoch_seen.add(gepoch)

            self.t_train.append(gepoch)
            for k, v in metrics.items():
                if k not in self.train_keys:
                    self.train_keys.append(k)
                self.train_series.setdefault(k, []).append(v)

        self._ensure_rollout_cb()
        self._ensure_train_cb()
        self._redraw()

    def _redraw(self):
        # rollout panel
        self.ax_rollout.cla()
        self.ax_rollout.set_title(f"Rollout metrics — ALL trials in {self.root.name}")
        self.ax_rollout.set_xlabel("global step")

        if self.t_reward and self.sel_rollout.get("reward", True):
            self.ax_rollout.plot(self.t_reward, self.reward_values, label="reward")

        for k in self.rollout_keys:
            if not self.sel_rollout.get(k, False):
                continue
            ys = self.rollout_series.get(k, [])
            xs = self.t_rollout[:len(ys)]
            if xs and ys:
                self.ax_rollout.plot(xs, ys, label=k)
        if self.ax_rollout.has_data():
            self.ax_rollout.legend(loc="upper left", fontsize=8)
            self.ax_rollout.relim(); self.ax_rollout.autoscale()

        # training panel
        self.ax_train.cla()
        self.ax_train.set_title("Training metrics — ALL trials")
        self.ax_train.set_xlabel("global epoch")
        for k in self.train_keys:
            if not self.sel_train.get(k, False):
                continue
            ys = self.train_series.get(k, [])
            xs = self.t_train[:len(ys)]
            if xs and ys:
                self.ax_train.plot(xs, ys, label=k)
        if self.ax_train.has_data():
            self.ax_train.legend(loc="upper left", fontsize=8)
            self.ax_train.relim(); self.ax_train.autoscale()

        self.fig.canvas.draw_idle()

# ---------------- cli ----------------
def main():
    ap = argparse.ArgumentParser(description="Live PPO Log Viewer (reward via trace_filter)")
    ap.add_argument("--root", type=str, required=True, help="Root folder containing trial_* subfolders")
    ap.add_argument("--control-config", type=str, required=True, help="Path to control_config JSON (reads reward_cfg)")
    ap.add_argument("--reward-agg", type=str, default="sum", choices=["sum", "mean"],
                    help="Aggregate flattened reward leaves by 'sum' (default) or 'mean'")
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh interval (s), >= 0.2")
    ap.add_argument("--rollout-keys", nargs="*", default=[], help="Preselect rollout metric keys")
    ap.add_argument("--train-keys", nargs="*", default=[], help="Preselect train metric keys")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERR] Root not found: {root}", file=sys.stderr)
        sys.exit(2)

    control_config_path = Path(args.control_config).resolve()
    if not control_config_path.exists():
        print(f"[ERR] control_config not found: {control_config_path}", file=sys.stderr)
        sys.exit(2)

    viewer = LivePlotAll(
        root=root,
        control_config_path=control_config_path,
        reward_agg=args.reward_agg,
        refresh=args.refresh,
        pre_rollout=args.rollout_keys,
        pre_train=args.train_keys,
    )

    print("[INFO] Viewing ALL trials under:", root)
    print("      Reward = trace_filter(record, reward_cfg) → flatten_leaves →", args.reward_agg)
    print("Controls:")
    print("  • Toggle series with the right-side checkboxes.")
    print("  • Press 'r' to autoscale axes.")
    plt.show()

if __name__ == "__main__":
    main()
