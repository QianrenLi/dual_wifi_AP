from __future__ import annotations
from typing import Dict, Iterable, Optional, Union, List, Tuple
from pathlib import Path
from collections import defaultdict, Counter

import re
import json

FolderRollout = lambda folder: folder / "rollout.jsonl" 
FolderRtt = lambda folder: folder / "rtt-6203@128.txt"
FolderChannel = lambda folder: folder / "output.log"

class Rollout:
    """
    Read one rollout JSONL file (one JSON per line).
    Expected structure per line (example shown by user):
    {
      "t": <float>, "iteration": <int>, "links": ["<lid>"],
      "action": { "<lid>": { ... }},            # optional (ignored here)
      "stats": {
        "flow_stat": { "<lid>": {"rtt": <f>, "outage_rate": <f>, "throughput": <f>, ... }},
        "device_stat": {
          "queues": { "<ip>": { "<q>": <int>, ... } }
        }
      },
      "res": {
        "action": [<floats>],
        "belief": [<floats>] or <float>
      }
    }

    What this class collects:
    - time      : list[float]
    - iteration : list[int]
    - rtt       : dict[link_id] -> list[float]
    - outage    : dict[link_id] -> list[float]
    - throughput: dict[link_id] -> list[float]
    - queues    : dict[ip] -> dict[queue_id] -> list[int]    (per-timestep)
    - queues_sum_by_ip : dict[ip] -> list[int]               (sum over that IP's queues per-step)
    - queues_sum_total : list[int]                           (sum over all IPs per-step)
    - action    : list[list[float]]                          (policy action vector per-step)
    - belief    : list[float] or list[list[float]]           (belief(s) per-step)
    """

    def __init__(self, file_path: Path, optional_handlers:Dict = {}):
        self.file_path = Path(file_path)

        # Scalars / simple series
        self.time: List[float] = []
        self.iteration: List[int] = []
        self.action: List[List[float]] = []
        self.belief: List[Union[float, List[float]]] = []

        # Per-link flow stats
        self.rtt: Dict[str, List[float]] = defaultdict(list)
        self.outage: Dict[str, List[float]] = defaultdict(list)
        self.throughput: Dict[str, List[float]] = defaultdict(list)

        # Queues
        self.queues: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.queues_sum_by_ip: Dict[str, List[int]] = defaultdict(list)
        self.queues_sum_total: List[int] = []
        
        self.optional_data = { key:[] for key in optional_handlers.keys()}

        self._load(optional_handlers)

    # ---------------------------- Public API ----------------------------
    def get_links(self) -> List[str]:
        """All link IDs that appeared in the file (order not guaranteed)."""
        return sorted(set(self.rtt.keys()) | set(self.outage.keys()) | set(self.throughput.keys()))

    def get_ips(self) -> List[str]:
        """All device IPs that appeared in device_stat.queues."""
        return sorted(self.queues.keys())

    def get_flow_series(self, link_id: str) -> Tuple[List[float], List[float], List[float]]:
        """Return (rtt, outage, throughput) for a given link_id."""
        return (self.rtt.get(link_id, []),
                self.outage.get(link_id, []),
                self.throughput.get(link_id, []))

    def get_queue_series(self, ip: str, queue_id: Optional[str] = None) -> Union[List[int], Dict[str, List[int]]]:
        """
        If queue_id is given, return that queue's series for the IP.
        Else return the dict of all queues for that IP.
        """
        if ip not in self.queues:
            return [] if queue_id is not None else {}
        return self.queues[ip].get(queue_id, []) if queue_id is not None else self.queues[ip]
    # --------------------------- Internal load --------------------------

    def _load(self, optional_handlers={}) -> None:
        with self.file_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # Skip malformed lines safely
                    continue

                self._parse_header(rec)
                self._parse_flow_stats(rec)
                self._parse_queues(rec)
                self._parse_result(rec)
                
                for key, handler in optional_handlers.items():
                    self.optional_data[key].append(handler(rec))

        # Normalize lengths for per-link series:
        self._pad_flow_series_to_len(len(self.time))
        
    # --------------------------- Parse helpers --------------------------

    def _parse_header(self, rec: dict) -> None:
        # t and iteration
        t = rec.get("t")
        if isinstance(t, (int, float)):
            self.time.append(float(t))
        else:
            # still keep indexes aligned; use None or -1
            self.time.append(float("nan"))

        itr = rec.get("iteration")
        self.iteration.append(int(itr) if isinstance(itr, int) else -1)

    def _parse_flow_stats(self, rec: dict) -> None:
        stats = rec.get("stats") or {}
        flow = (stats.get("flow_stat") or {})
        links: Iterable[str] = rec.get("links") or flow.keys()

        # For each link that shows up, try to read rtt/outage/throughput; if missing, pad with NaN
        for lid in links:
            fs = flow.get(lid, {})
            self.rtt[lid].append(self._to_float(fs.get("rtt")))
            self.outage[lid].append(self._to_float(fs.get("outage_rate")))
            self.throughput[lid].append(self._to_float(fs.get("bitrate")))

        # Also ensure that previously seen links (but not present on this line) are padded
        for lid in self.rtt.keys():
            if lid not in links:
                self._ensure_pad_missing(lid)

    def _parse_queues(self, rec: dict) -> None:
        stats = rec.get("stats") or {}
        dev = stats.get("device_stat") or {}
        qroot = dev.get("queues") or {}

        # Accumulate per-queue series; if an IP doesn't appear this step, pad with previous step's "missing" (0)
        ips_seen = set()
        step_total = 0

        for ip, qdict in qroot.items():
            ips_seen.add(ip)
            # Make sure all prior-known queue IDs for this IP get something this step
            known_qids = set(self.queues[ip].keys()) | set(qdict.keys())
            ip_sum = 0
            for qid in known_qids:
                val = int(qdict.get(qid, 0))
                self.queues[ip][qid].append(val)
                ip_sum += val
            self.queues_sum_by_ip[ip].append(ip_sum)
            step_total += ip_sum

        # Pad missing IPs with zeros at this step
        for ip in list(self.queues.keys()):
            if ip not in ips_seen:
                # Pad all known queues of this IP with 0
                ip_sum = 0
                for qid in list(self.queues[ip].keys()):
                    self.queues[ip][qid].append(0)
                self.queues_sum_by_ip[ip].append(0)

        self.queues_sum_total.append(step_total)

    def _parse_result(self, rec: dict) -> None:
        res = rec.get("res") or {}

        # action vector
        act = res.get("action")
        if isinstance(act, list):
            # Enforce float conversion
            self.action.append([float(x) for x in act])
        else:
            self.action.append([])

        # belief can be a list or scalar—preserve as list[float] if list, else float
        belief = res.get("belief")
        if isinstance(belief, list):
            self.belief.append(float(belief[0]))
        elif isinstance(belief, (int, float)):
            self.belief.append(float(belief))
        else:
            self.belief.append(float("nan"))

    # ----------------------------- Utilities ----------------------------

    @staticmethod
    def _to_float(x: object) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _ensure_pad_missing(self, link_id: str) -> None:
        """
        If a link was seen before but not in the current line, pad its series
        so all links keep the same length as time/iteration.
        """
        target_len = len(self.time)
        for series in (self.rtt[link_id], self.outage[link_id], self.throughput[link_id]):
            if len(series) < target_len:
                series.append(float("nan"))

    def _pad_flow_series_to_len(self, L: int) -> None:
        """After the pass, ensure all per-link series have length L."""
        for lid in set(self.rtt.keys()) | set(self.outage.keys()) | set(self.throughput.keys()):
            for series in (self.rtt[lid], self.outage[lid], self.throughput[lid]):
                while len(series) < L:
                    series.append(float("nan"))
    
class RttLog:
    """
    Reads an RTT log where each non-empty line is:
        <seq_id:int> <rtt_in_seconds:float> [<timestamp:float>]

    - Lines may have 2 or 3 columns; the 3rd (timestamp) is ignored.
    - Blank lines and lines starting with '#' are skipped.
    - Malformed lines are safely ignored.
    """
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._seq_ids: List[int] = []
        self._rtts: List[float] = []
        self._load()

    def _load(self) -> None:
        with self.file_path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    # Not enough columns; skip
                    continue
                try:
                    seq = int(parts[0])
                    rtt = float(parts[1])
                except ValueError:
                    # Bad number format; skip
                    continue
                self._seq_ids.append(seq)
                self._rtts.append(rtt)

    # --- API ---
    def get_rtts(self) -> List[float]:
        """Return the RTTs in seconds, in file order."""
        return self._rtts

    def get_seq_ids(self) -> List[int]:
        """Return the sequence IDs, in file order."""
        return self._seq_ids

    def as_lists(self) -> Tuple[List[int], List[float]]:
        """Convenience method returning (seq_ids, rtts)."""
        return self._seq_ids, self._rtts

class ChannelLog:
    """
    Parse lines like:
        INFO - 0, 1
        INFO - 1, 3
    where the first number is interface id (0/1) and the second is the seq id.
    """

    INFO_RE = re.compile(r"INFO\s*-\s*(\d+)\s*,\s*(\d+)\b")

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        # Raw per-seq sightings (ordered as they appear)
        self._seq_to_ifaces: Dict[int, List[int]] = defaultdict(list)
        self._load()

    # ---------------- API (per your new definitions) ----------------
    def get_seq_list(self) -> List[int]:
        """Sorted list of unique seq ids observed."""
        return sorted(self._seq_to_ifaces.keys())

    def get_interface_percentages(self, group_factor: Optional[int] = None) -> Dict[int, List[float]]:
        """
        If group_factor is None:
            Per-seq percentages:
                { seq_id: [pct_if0, pct_if1] }
            Percentages are w.r.t. that seq's own sightings.

        If group_factor is an int > 0:
            Per-group percentages:
                { group_id: [pct_if0, pct_if1] }, where group_id in [0, group_factor-1]
            Sequences are bucketed by seq_id % group_factor.
            Percentages are computed from the group's total sightings (weighted by counts).
        """
        if group_factor is None:
            result: Dict[int, List[float]] = {}
            for seq, ifaces in self._seq_to_ifaces.items():
                n = len(ifaces)
                if n == 0:
                    result[seq] = [0.0, 0.0]
                    continue
                c = Counter(ifaces)
                pct0 = 100.0 * c.get(0, 0) / n
                pct1 = 100.0 * c.get(1, 0) / n
                result[seq] = [pct0, pct1]
            return result

        # ---- grouped mode ----
        if group_factor <= 0:
            raise ValueError("group_factor must be a positive integer")

        # totals[group_id] = [count_if0, count_if1, total_sightings]
        totals = defaultdict(lambda: [0, 0, 0])
        for seq, ifaces in self._seq_to_ifaces.items():
            n = len(ifaces)
            if n == 0:
                continue
            g = int(seq) // group_factor
            c = Counter(ifaces)
            c0 = c.get(0, 0)
            c1 = c.get(1, 0)
            totals[g][0] += c0
            totals[g][1] += c1
            totals[g][2] += n

        grouped: Dict[int, List[float]] = {}
        for g in totals.keys():
            n0, n1, N = totals.get(g, [0, 0, 0])
            if N == 0:
                grouped[g] = [0.0, 0.0]
            else:
                grouped[g] = [100.0 * n0 / N, 100.0 * n1 / N]
        return grouped

    def get_num_seqs(self) -> Dict[int, int]:
        """
        Per-seq packet counts (number of sightings/lines per seq):
            { seq_id: count }
        """
        return {seq: len(ifaces) for seq, ifaces in self._seq_to_ifaces.items()}

    # Optional overall helpers if you still want them:
    def get_overall_interface_percentages(self) -> List[float]:
        """
        Overall percentages across all sightings (not per seq):
        returns [pct_if0, pct_if1].
        """
        all_ifaces = [iface for ifs in self._seq_to_ifaces.values() for iface in ifs]
        n = len(all_ifaces)
        if n == 0:
            return [0.0, 0.0]
        c = Counter(all_ifaces)
        return [100.0 * c.get(0, 0) / n, 100.0 * c.get(1, 0) / n]

    def get_seq_to_interface_majority(self) -> Dict[int, int]:
        """
        Majority assignment per seq (tie → last seen).
        Not requested, but often useful.
        """
        assigned: Dict[int, int] = {}
        for seq, ifaces in self._seq_to_ifaces.items():
            c = Counter(ifaces)
            if c.get(0, 0) > c.get(1, 0):
                assigned[seq] = 0
            elif c.get(1, 0) > c.get(0, 0):
                assigned[seq] = 1
            else:
                assigned[seq] = ifaces[-1] if ifaces else 0
        return assigned

    # ---------------- Internal ----------------
    def _load(self) -> None:
        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                m = self.INFO_RE.search(line)
                if not m:
                    continue
                iface_s, seq_s = m.groups()
                try:
                    iface = int(iface_s)
                    seq = int(seq_s)
                except ValueError:
                    continue
                if iface not in (0, 1):
                    continue
                self._seq_to_ifaces[seq].append(iface)


class ExpTraceReader:
    # ------------------------ Folder / naming utils ------------------------ #
    _IL_PATTERN = re.compile(r"^IL_(\d+)_trial_(.+)$")
    
    def __init__(self, meta_folder: Path):
        self.meta_folder = Path(meta_folder) if isinstance(meta_folder, str) else meta_folder
        self.latest_folders = self.pick_latest_per_il()

    def extract_il_and_trial(self, name: str) -> Tuple[Optional[int], Optional[str]]:
        m = self._IL_PATTERN.match(name)
        if not m:
            return None, None
        try:
            return int(m.group(1)), m.group(2)
        except Exception:
            return None, m.group(2)

    def pick_latest_per_il(
        self, 
        pattern: str = "IL_*_trial_*",
        by_mtime: bool = True) -> Dict[int, Path]:
        """
        Scan `meta_folder` for IL_*_trial_* subfolders, group by IL id,
        and pick the latest (by mtime or lexicographic name) per IL.
        Returns { il_id: latest_run_folder_path }.
        """
        if not self.meta_folder.is_dir():
            return {}
        runs = [p for p in self.meta_folder.glob(pattern) if p.is_dir()]
        groups: Dict[int, List[Path]] = {}
        for p in runs:
            il_id, _ = self.extract_il_and_trial(p.name)
            if il_id is None:
                continue
            groups.setdefault(il_id, []).append(p)

        latest: Dict[int, Path] = {}
        for il_id, folders in groups.items():
            if not folders:
                continue
            if by_mtime:
                folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            else:
                folders.sort(key=lambda x: x.name, reverse=True)
            latest[il_id] = folders[0]
        return latest
    
    def pick_last_k_per_il(
        self,
        k: int,
        pattern: str = "IL_*_trial_*",
        by_mtime: bool = True
    ) -> Dict[int, List[Path]]:
        if k < 0 or not self.meta_folder.is_dir():
            return {}
        runs = [p for p in self.meta_folder.glob(pattern) if p.is_dir()]
        groups: Dict[int, List[Path]] = {}
        for p in runs:
            il_id, _ = self.extract_il_and_trial(p.name)
            if il_id is None:
                continue
            groups.setdefault(il_id, []).append(p)

        result: Dict[int, List[Path]] = {}
        for il_id, folders in groups.items():
            if not folders:
                continue
            if by_mtime:
                folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            else:
                folders.sort(key=lambda x: x.name, reverse=True)
            result[il_id] = folders[k]
        return result

    def get_data(self, run_folder: Path) -> Tuple[Optional[Rollout], Optional[RttLog], Optional[ChannelLog]]:
        rollout     = FolderRollout(run_folder) 
        rtt         = FolderRtt(run_folder)
        interface   = FolderChannel(run_folder)
        
        rollout     = Rollout(rollout) if rollout.exists() else None
        rtt         = RttLog(rtt) if rtt.exists() else None
        interface   = ChannelLog(interface) if interface.exists() else None
    
        return rollout, rtt, interface
    
    