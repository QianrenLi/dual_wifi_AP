# util/trace_collec.py
from __future__ import annotations

import json
import logging
import numpy as np
from typing import Any, Callable, Dict, Optional, Union, Mapping, Tuple, List, Iterable

from util.filter_rule import FILTER_REGISTRY

logger = logging.getLogger(__name__)

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


def create_obs(
    state: List,
    action_dim: int,
    action: List | None,
):
    if action is not None:
        state.extend(action)
    else:
        state.extend([0.0] * action_dim)
    return state
    
    
def create_obss(
    states: List[List],
    actions: List[List],
    rewards: List[List],
) -> List[List]:
    """
    Insert previous-step action/reward into each state:
      s'[t] = [ s[t], a[t-1], r[t-1] ] with a[-1]=0, r[-1]=0.
    If the state already has reserved tail slots of size action_dim+reward_dim
    (commonly zeros), they are filled in-place; otherwise the vectors are appended.

    Returns a list of *lists* (flattened obs per step).
    """
    if not states:
        return []

    T = len(states)
    assert len(actions) == T and len(rewards) == T, "states/actions/rewards length mismatch"

    action_dim = len(actions[0])

    out: List[List] = []
    for t in range(T):
        action = actions[t-1] if t > 0 else None
        out.append(create_obs(states[t], action_dim, action))

    return out

def trace_collec(
    json_file: str,
    state_descriptor: Optional[Descriptor] = None,
    reward_descriptor: Optional[Descriptor] = None
):
    trace_items = []
    with open(json_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trace_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse JSON in {json_file}:{line_num}: {e.msg} "
                    f"(column {e.colno}, char {e.pos}). Skipping this line. "
                    f"Context: ...{line[max(0, e.pos-30):min(len(line), e.pos+30)]}..."
                )
                continue

    if not trace_items:
        logger.warning(f"No valid trace items found in {json_file}")
        return [], [], [], []

    actions = [flatten_leaves(t.get("res").get("action")) for t in trace_items]
    states  = [flatten_leaves(trace_filter(t, state_descriptor))  for t in trace_items]
    rewards = [flatten_leaves(trace_filter(t, reward_descriptor)) for t in trace_items]
    network_output = [t.get("res") for t in trace_items]

    states = create_obss(states=states, actions=actions, rewards=rewards)

    states = states[:-1]
    actions = actions[:-1]
    network_output = network_output[:-1]
    rewards = rewards[1:]

    return states, actions, rewards, network_output

# ============================== Demo ============================= #
if __name__ == "__main__":
    STATE_DESCRIPTOR = {
        "signal_dbm": { "rule": True, "name": "signal_dbm", "pos": "agent" },
        "tx_mbit_s":  { "rule": True, "name": "tx_mbit_s",  "pos": "agent" },

        "queues":     { "rule": "queues_only_ac1", "from": "queues", "pos": "agent" },
        
        "app_buff":    { "rule": True, "from": "app_buff", "pos": "flow" },
        "frame_count": { "rule": True, "from": "frame_count", "pos": "flow" },
        "left_frame":  { "rule": True, "from": "left_frame", "pos": "flow" },
            
        "bitrate": {
            "rule": True,
            "from": "bitrate",
            "pos": "flow"
        },
        "outage_rate": { "rule": True, "from": "outage_rate", "pos": "flow" },
        
        "acc_bitrate": { "rule": True, "from": "acc_bitrate", "pos": "flow" },
        "diff_bitrate": { "rule": True, "from": "diff_bitrate", "pos": "flow" }
    }
    
    REWARD_DESCRIPTOR = {
        "acc_bitrate": {
            "rule": "stat_bitrate",
            "from": "acc_bitrate",                 # bare key -> flow["bitrate"]
            "args": {"alpha": 1e-6 / 5},
            "pos": "flow",
        },
        "diff_bitrate": {
            "rule": "stat_bitrate",
            "from": "diff_bitrate",               # bare key -> flow["bitrate"]
            "args": {"alpha": -1e-6 / 2 / 5}, 
            "pos": "flow",
        },
        "outage_rate": {
            "rule": "scale_outage",
            "from": "outage_rate",             # bare key -> flow["outage_rate"]
            "args": {"zeta": -50 / 5 },
            "pos": "flow",
        }
    }
    
    
    
    # s, a, r, n = trace_collec( '/home/qianren/workspace/dual_wifi_AP/exp_trace/rnn_1500_day/IL_0_trial_20251114-103432/rollout.jsonl' ,state_descriptor=STATE_DESCRIPTOR, reward_descriptor=REWARD_DESCRIPTOR)
    # print(s[0])
    # print(len(s[0]))
    
    # python compute_normalization.py --control_config net_util/net_config/rnn_1500_day/test.json --trace_path exp_trace/rnn_1500_day
    
    #################
    example_js_str = '''
{"t": 1764925932.6989014, "iteration": 0, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM1_0_1", "__data__": [0.0, 0.0, 0.0, 0.006987065076828003]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 12}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0025925318400065104, "outage_rate": 0.0, "throughput": 2.31693872149037, "throttle": 0.0, "bitrate": 2000000, "acc_bitrate": 2000000, "diff_bitrate": 0, "app_buff": 1, "frame_count": 2, "left_frame": 47}}, "device_stat": {"taken_at": {"secs_since_epoch": 1764925932, "nanos_since_epoch": 694822594}, "queues": {"192.168.3.35": {"1": 5, "2": 0, "0": 0}, "192.168.3.25": {"0": 0, "2": 0}}, "link": {"192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5220, "signal_dbm": -32, "tx_mbit_s": 867.0}, "192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -38, "tx_mbit_s": 174.0}}}}, "timed_out": false, "res": {"action": [-0.986025869846344, 0.9945688843727112], "log_prob": [7.842179775238037], "value": 0, "belief": [0.9139199256896973]}, "policy": "SACRNNBeliefSeqDistV9"}
    '''
    example_js_str_2 = '''
    {"t": 1764925932.7029114, "iteration": 1, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM1_0_1", "__data__": [0.0, 0.0, 0.0, 0.03185001015663147]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 12}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0, "outage_rate": 0.0, "throughput": 2.31693872149037, "throttle": 0.0, "bitrate": 2000000, "acc_bitrate": 0, "diff_bitrate": 0, "app_buff": 0, "frame_count": 0, "left_frame": 47}}, "device_stat": {"taken_at": {"secs_since_epoch": 1764925932, "nanos_since_epoch": 700517411}, "queues": {"192.168.3.35": {"1": 0, "2": 0, "0": 0}, "192.168.3.25": {"0": 0, "2": 0}}, "link": {"192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5220, "signal_dbm": -32, "tx_mbit_s": 867.0}, "192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -38, "tx_mbit_s": 174.0}}}}, "timed_out": false, "res": {"action": [-0.9362999796867371, 0.8775613307952881], "log_prob": [1.0470383167266846], "value": 0, "belief": [0.9304964542388916]}, "policy": "SACRNNBeliefSeqDistV9"}
    '''
    
    
    ###################
    print(trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR))
    print(flatten_leaves(trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR)))
    print(flatten_leaves(trace_filter(json.loads(example_js_str.replace("'", '"')), REWARD_DESCRIPTOR)))
    exit()
    # print(trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR))
    # print(flatten_leaves(trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR)))
    
    # states = [ trace_filter(json.loads(example_js_str2.replace("'", '"')), STATE_DESCRIPTOR),  trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR)]
    
    ##################
    # test = create_obs(states)[0]


    #########
    STATE_TRANSFORM = {'created_at': '2025-11-14 15:14:42',
    'dim': 13,
    'meta': {'control_config': '/home/qianren/workspace/dual_wifi_AP/net_util/net_config/rnn_1500_day/test.json',
            'trace_path': '/home/qianren/workspace/dual_wifi_AP/exp_trace/rnn_1500_day'},
    'state': {'mean': [-38.76929473876953,
                        -30.3992862701416,
                        175.18724060058594,
                        864.6111450195312,
                        1.1262472867965698,
                        2.1851847171783447,
                        0.23867054283618927,
                        0.008045737631618977,
                        30,
                        19276342.0,
                        0.12414762377738953,
                        19276342.0,
                        19276342.0,
                        0.0,
                        0.0],
            'std': [1.9944250583648682,
                    3.2788290977478027,
                    1.1888312101364136,
                    2.3938241004943848,
                    5.041040420532227,
                    7.8146562576293945,
                    5.00417423248291,
                    0.09046436846256256,
                    3.46,
                    9105042.0,
                    7.309142112731934,
                    9105042.0,
                    9105042.0,
                    1.0,
                    1.0]}}



    from net_util.state_transfom import _StateTransform
    import time
    _state_tf = _StateTransform.from_obj(STATE_TRANSFORM)
    
    start_time = time.time()
    jss = [json.loads(example_js_str.replace("'", '"')), json.loads(example_js_str_2.replace("'", '"'))]
    state_value = create_obss(
        states=[flatten_leaves(trace_filter(t, STATE_DESCRIPTOR)) for t in jss],
        actions=[flatten_leaves(t.get("res").get("action")) for t in jss],
        rewards=[flatten_leaves(trace_filter(t, REWARD_DESCRIPTOR)) for t in jss],
    )
    print(state_value[0])
    print(_state_tf.apply_to_list(state_value[0]))
    print(state_value[1])
    print(_state_tf.apply_to_list(state_value[1]))
    print("time cost:", time.time() - start_time)
    