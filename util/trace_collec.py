# util/trace_collec.py
from __future__ import annotations

import json
import numpy as np
from typing import Any, Callable, Dict, Optional, Union, Mapping, Tuple, List, Iterable

from util.control_cmd import _json_object_hook
from util.filter_rule import FILTER_REGISTRY

# ----------------------------- Types ----------------------------- #
Rule = Union[bool, Callable[[Any], Any], str]
DescriptorEntry = Dict[str, Any]   # supports: rule, from, name, args, pos (reserved)
Descriptor = Dict[str, DescriptorEntry]

# ======================= Lightweight helpers ===================== #
def _is_ipv4(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 and str(int(p)) == p for p in parts)
    except ValueError:
        return False

def _ip_key(ip: str) -> Tuple[int, int, int, int]:
    return tuple(int(x) for x in ip.split("."))  # type: ignore[return-value]

def _all_digit_keys(keys: Iterable[str]) -> bool:
    try:
        return all(str(int(k)) == k for k in keys)
    except Exception:
        return False

def flatten_leaves(struct: Any) -> List[float]:
    """Collect leaf numeric values; fallback to .value(list) if present.
    NOTE: 保留原有 print 行为以保证兼容。"""
    out: List[float] = []
    if isinstance(struct, dict):
        for v in struct.values():
            out.extend(flatten_leaves(v))
    elif isinstance(struct, (list, tuple)):
        for v in struct:
            out.extend(flatten_leaves(v))
    else:
        try:
            out.append(float(struct))
        except (ValueError, TypeError):
            value = getattr(struct, "value", None)
            if isinstance(value, list):
                out.extend(value)
            else:
                # 保持原诊断输出
                print(f"leave {value} is not added")
        except Exception:
            pass
    return out

def struct_to_numpy(struct: Any) -> np.ndarray:
    return np.array(flatten_leaves(struct), dtype=float)

# ============================ Path utils ========================= #
def _split_path(p: str) -> Tuple[str, ...]:
    return tuple(seg for seg in p.split(".") if seg)

def _get_by_path(root: Any, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    cur = root
    for seg in path:
        if not isinstance(cur, dict) or seg not in cur:
            return False, None
        cur = cur[seg]
    return True, cur

# ======================== Source selectors ======================= #
def _pick_first_mapping_value(m: Mapping) -> Any:
    for _, v in m.items():
        return v
    return None

def _get_flow_map(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stats = trace.get("stats", {})
    flows = stats.get("flow_stat", {})
    return flows if isinstance(flows, dict) and flows else None

def _choose_flow_node(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Prefer the first link id in 'links', else the first item in stats.flow_stat."""
    flows = _get_flow_map(trace)
    if not flows:
        return None
    links = trace.get("links")
    if isinstance(links, list) and links:
        node = flows.get(links[0])
        if isinstance(node, dict):
            return node
    return _pick_first_mapping_value(flows)

def _get_agent_link_map(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the FULL ip->link dict under device_stat.link."""
    stats = trace.get("stats", {})
    dev  = stats.get("device_stat", {})
    link = dev.get("link")
    return link if isinstance(link, dict) and link else None

def _get_agent_queues_map(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the FULL ip->queues mapping under device_stat.queues."""
    stats = trace.get("stats", {})
    dev   = stats.get("device_stat", {})
    q = dev.get("queues")
    return q if isinstance(q, dict) and q else None

# ====================== Context search helpers =================== #
def _search_value_in_contexts(trace: Dict[str, Any], key: str) -> Any:
    """Find a value/dict by key across common contexts in a deterministic order."""
    # 1) root
    if isinstance(trace, dict) and key in trace:
        return trace[key]
    # 2) flow
    flow = _choose_flow_node(trace)
    if isinstance(flow, dict) and key in flow:
        return flow[key]
    # 3) agent.link (try full map first)
    link_map = _get_agent_link_map(trace)
    if isinstance(link_map, dict) and key in link_map:
        return link_map[key]
    # 4) queues special-case: return FULL mapping
    if key == "queues":
        qmap = _get_agent_queues_map(trace)
        if isinstance(qmap, dict):
            return qmap
    return None

def _get_by_outkey_default(trace: Dict[str, Any], out_key: str) -> Any:
    """Original behavior default source: out_key at root (supports dotted)."""
    if "." in out_key:
        ok, node = _get_by_path(trace, _split_path(out_key))
        return node if ok else None
    return trace.get(out_key)

# ================ Fan-out extraction over child dicts ============ #
def _extract_named_from_children(node: Any, name: str) -> Any:
    """
    If node is a dict[str -> dict], return dict[str -> child[name]] (when present),
    preserving key order. Else, return None.
    """
    if not isinstance(node, dict):
        return None
    out: Dict[str, Any] = {}
    for k, v in node.items():
        if isinstance(v, dict) and name in v:
            out[k] = v[name]
    return out if out else None

# ========================== Rule application ====================== #
def _apply_rule(value: Any, rule: Rule, args: Optional[dict]) -> Any:
    if rule is True:
        return value
    if rule is False or rule is None:
        return None
    if isinstance(rule, str):
        func = FILTER_REGISTRY.get(rule)
        if func is None:
            raise AssertionError(f"Unknown filter '{rule}' not in FILTER_REGISTRY")
        return func(value, **(args or {}))
    if callable(rule):
        return rule(value, **(args or {}))
    return None

# ============================ Sorting ============================ #
def _deep_sort_mappings(node: Any) -> Any:
    """Sort only when keys are IPv4 or pure digit strings; otherwise keep insertion order."""
    if isinstance(node, dict):
        items = [(k, _deep_sort_mappings(v)) for k, v in node.items()]
        keys = [k for k, _ in items]
        if keys and all(isinstance(k, str) for k in keys):
            if all(_is_ipv4(k) for k in keys):
                items.sort(key=lambda kv: _ip_key(kv[0]))
            elif _all_digit_keys(keys):
                items.sort(key=lambda kv: int(kv[0]))
        return {k: v for k, v in items}
    if isinstance(node, list):
        return [_deep_sort_mappings(x) for x in node]
    return node

# ============================ Resolver =========================== #
def _resolve_source(trace: Dict[str, Any], entry: DescriptorEntry, out_key: str) -> Any:
    """
    Resolve the data 'source' for a descriptor entry.

    Priority:
      A) If entry["from"] is provided:
         - special tokens: root|flow|agent|agent.link|agent.queues|queues|res|action
         - dotted path from root: e.g. "res.action"
         - bare key: search across contexts (root -> flow -> agent.link -> queues-special)
         NOTE: for 'agent'/'agent.link' contexts we return the FULL map.
      B) If 'from' is omitted:
         - original behavior by out_key (root/dotted)
         - if not found and 'name' provided: prefer FULL agent.link map, fanning out by 'name'
         - else: search out_key across contexts

    'pos' is reserved/ignored.
    """
    src  = entry.get("from")
    name = entry.get("name")

    # -------- A) explicit 'from' provided --------
    if isinstance(src, str) and src:
        if src == "root":
            node = trace
        elif src == "flow":
            node = _choose_flow_node(trace)
        elif src in ("agent", "agent.link"):
            node = _get_agent_link_map(trace)        # FULL link map
        elif src in ("agent.queues", "queues"):
            node = _get_agent_queues_map(trace)      # FULL queues map
        elif src == "res":
            node = trace.get("res")
        elif src == "action":
            node = trace.get("action")
        elif "." in src:
            ok, node = _get_by_path(trace, _split_path(src))
            node = node if ok else None
        else:
            # bare key -> search across contexts
            node = _search_value_in_contexts(trace, src)

        # If we got a full link map and a 'name', fan-out into per-IP mapping
        if name and isinstance(node, dict):
            fan = _extract_named_from_children(node, name)
            if fan is not None:
                return fan

        # Otherwise, if 'name' is present and node is dict, try subkey
        if name is not None and isinstance(node, dict):
            node = node.get(name)
        return node

    # -------- B) 'from' omitted --------
    # Original behavior
    node = _get_by_outkey_default(trace, out_key)

    # If not found and name exists, prefer full agent.link map and fan-out by name
    if node is None and name:
        link_map = _get_agent_link_map(trace)
        if isinstance(link_map, dict):
            fan = _extract_named_from_children(link_map, name)
            if fan is not None:
                return fan
        # Otherwise search by name across contexts
        node = _search_value_in_contexts(trace, name)

    if node is None:
        node = _search_value_in_contexts(trace, out_key)

    return node

# ============================ Public API ========================= #
def trace_filter(trace: Any, descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Any:
    """
    Apply descriptor in order (dict order preserved).

    Each entry supports:
      {
        "rule": True|False|callable|rule_name,   # required (default True)
        "from": "<source>" | dotted.path,        # optional; if absent, use descriptor key as source
        "name": "<subkey>",                      # optional; pick subkey / or fan-out over agent.link map
        "args": {...},                           # optional; passed to rule
        "pos": "...",                            # reserved; ignored by the engine
      }

    No 'copied' support.
    """
    if not isinstance(trace, dict) or not isinstance(descriptor, Mapping) or not descriptor:
        return trace

    out: Dict[str, Any] = {}
    for out_key, entry in descriptor.items():  # preserves insertion order
        if not isinstance(entry, Mapping):
            # Shorthand: entry itself is the rule, source is the same-named key (original behavior)
            rule: Rule = entry  # type: ignore[assignment]
            src_node = _resolve_source(trace, {}, out_key)
            val = _apply_rule(src_node, rule, None)
        else:
            rule: Rule = entry.get("rule", True)
            args = entry.get("args")
            # 'pos' is reserved — do nothing with it
            src_node = _resolve_source(trace, entry, out_key)

            # If we have a full per-IP dict and rule is True, we can return it directly.
            # (Already handled fan-out above when 'name' present.)
            val = _apply_rule(src_node, rule, args)

        if val is None:
            continue
        if isinstance(val, dict) and not val:
            continue
        out[out_key] = val

    return _deep_sort_mappings(out)  # Only sorts IPv4/digit-keyed dicts; preserves other insertion orders

def shift_res_action_in_states(
    states: List[Dict[str, Any]],
    path: Tuple[str, ...] = ("res_action",),
) -> List[Dict[str, Any]]:
    """把 states 中 path 指向的向量整体右移 1（首个置零），仅当路径已存在时写入。"""
    # Getter
    def _get(d: Dict[str, Any]) -> Tuple[bool, Any]:
        return _get_by_path(d, path)

    # In-place setter (only if existing)
    p = path
    last = p[-1]
    pre = p[:-1]
    def _set_if_exists(d: Dict[str, Any], value: Any) -> bool:
        cur: Any = d
        for seg in pre:
            if not isinstance(cur, dict) or seg not in cur or not isinstance(cur[seg], dict):
                return False
            cur = cur[seg]
        if not isinstance(cur, dict) or last not in cur:
            return False
        cur[last] = value
        return True

    n: Optional[int] = None
    for s in states:
        ok, a = _get(s)
        if ok and isinstance(a, (list, tuple)) and a:
            n = len(a)
            break
    if not n:
        return states

    prev = [0.0] * n
    for s in states:
        ok, cur = _get(s)
        wrote = _set_if_exists(s, list(prev))
        if wrote and ok and isinstance(cur, (list, tuple)) and len(cur) == n:
            prev = list(cur)
    return states

def trace_collec(
    json_file: str,
    state_descriptor: Optional[Descriptor] = None,
    reward_descriptor: Optional[Descriptor] = None
):
    with open(json_file, "r") as f:
        trace_items = [json.loads(line, object_hook=_json_object_hook) for line in f]

    actions = [t.get("action") for t in trace_items]
    states  = [trace_filter(t, state_descriptor)  for t in trace_items]
    rewards = [trace_filter(t, reward_descriptor) for t in trace_items]
    network_output = [t.get("res") for t in trace_items]

    # in-place shift of res.action (+1) inside states
    states = shift_res_action_in_states(states, path=("rnn", "res", "action"))
    return states, actions, rewards, network_output

# ============================== Demo ============================= #
if __name__ == "__main__":
    STATE_DESCRIPTOR: Descriptor = {
        # agent-level features (choose deterministic IP under device_stat.link)
        "signal_dbm": { "rule": True, "name": "signal_dbm", "pos": "agent" },
        "tx_mbit_s":  { "rule": True, "name": "tx_mbit_s",  "pos": "agent" },

        # queues: pass the FULL ip->queues map to the rule
        "queues":     { "rule": "queues_only_ac1", "from": "queues", "pos": "agent" },

        # flow-level stats
        "bitrate":     { "rule": "stat_bitrate", "from": "bitrate", "args": {"alpha": 1e-6}, "pos": "flow" },
        
        "app_buff":    { "rule": True, "from": "app_buff", "pos": "flow" },
        "frame_count": { "rule": True, "from": "frame_count", "pos": "flow" },
        "rtt":         { "rule": True, "from": "rtt", "pos": "flow" },
        "outage_rate": { "rule": True, "from": "outage_rate", "pos": "flow" },

        # dotted path example
        "res_action":  { "rule": True, "from": "res.action", "pos": "flow" , "number": 5},
        
        "r_bitrate": {
            "rule": "bitrate_delta",
            "from": "bitrate",                 # bare key -> flow["bitrate"]
            "args": {"alpha": 1e-6, "beta": -1e-6 / 2},
            "pos": "flow",
        },
        "r_outage_rate": {
            "rule": "scale_outage",
            "from": "outage_rate",             # bare key -> flow["outage_rate"]
            "args": {"zeta": -1e3},
            "pos": "flow",
        },
    }

    example_js_str = '''
    {"t": 1760702113.036341, "iteration": 0, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [16.354024410247803, 40.01936316490173, 498.2471615076065, 107.10836946964264]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 11}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0048943062623341875, "outage_rate": 0.004726814726988474, "throughput": 7.396829444090822, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760702113, "nanos_since_epoch": 30374143}, "queues": {"192.168.3.25": {"1": 3, "0": 0, "2": 0}, "192.168.3.35": {"0": 0, "1": 0, "2": 0}}, "link": {"192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}, "192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -57, "tx_mbit_s": 174.0}}}}, "timed_out": false, "policy": "SAC"}
    '''
    
    example_js_str2 = '''
    {"t": 1760702113.036341, "iteration": 0, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [16.354024410247803, 40.01936316490173, 498.2471615076065, 107.10836946964264]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 11}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0048943062623341875, "outage_rate": 0.004726814726988474, "throughput": 7.396829444090822, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760702113, "nanos_since_epoch": 30374143}, "queues": {"192.168.3.35": {"1": 3, "0": 0, "2": 0}, "192.168.3.25": {"0": 0, "1": 0, "2": 0}}, "link": {"192.168.3.25": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}, "192.168.3.35": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -57, "tx_mbit_s": 174.0}}}}, "timed_out": false, "res": {"action": [1111, -0.7683377265930176, -0.9283580183982849, 0.911644458770752, 0.8903278708457947], "log_prob": [-16.09775161743164], "value": 0}, "policy": "SAC"}
    '''
    
    example_js_str3 = '''
    {"t": 1760690575.639269, "iteration": 2, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [441.0734474658966, 57.915568351745605, 17.910495400428772, 477.911114692688]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 12}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.004517078399658203, "outage_rate": 0.0, "throughput": 7.753064995008742, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760690575, "nanos_since_epoch": 638260169}, "queues": {"192.168.3.35": {"1": 0, "0": 0, "2": 0}, "192.168.3.25": {"2": 0, "0": 0, "1": 0}}, "link": {"192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -58, "tx_mbit_s": 174.0}, "192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}}}}, "timed_out": false, "res": {"action": [0.7642937898635864, -0.7683377265930176, -0.9283580183982849, 0.911644458770752, 0.8903278708457947], "log_prob": [-16.09775161743164], "value": 0}, "policy": "SAC"}
    '''
    
    print(trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR))
    print(flatten_leaves(trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR)))
    print(trace_filter(json.loads(example_js_str2.replace("'", '"')), STATE_DESCRIPTOR))
    print(flatten_leaves(trace_filter(json.loads(example_js_str2.replace("'", '"')), STATE_DESCRIPTOR)))
    print(trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR))
    print(len(flatten_leaves(trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR))))
    
    

    states = [ trace_filter(json.loads(example_js_str2.replace("'", '"')), STATE_DESCRIPTOR),  trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR)]
    
    # import copy
    
    # test_state = copy.deepcopy(states[1])
    test = shift_res_action_in_states(states)[0]
    
    print(test)
    # print(test_state)
    # print(states[1])
    