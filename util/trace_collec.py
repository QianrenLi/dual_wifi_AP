import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, Mapping, Tuple, List, Set
from util.control_cmd import _json_object_hook
from util.filter_rule import FILTER_REGISTRY

Rule = Union[bool, Callable[[Any], Any], Dict[str, "Rule"]]
DescriptorEntry = Dict[str, Any]
Descriptor = Dict[str, DescriptorEntry]

# ---------- Unchanged public helper ----------

def flatten_leaves(struct: Any) -> List[float]:
    """Recursively collect all leaf values (numbers) from a nested dict/list structure."""
    leaves: List[float] = []

    if isinstance(struct, dict):
        for v in struct.values():
            leaves.extend(flatten_leaves(v))
    elif isinstance(struct, (list, tuple)):
        for v in struct:
            leaves.extend(flatten_leaves(v))
    else:
        # treat as leaf
        try:
            leaves.append(float(struct))
        except (ValueError, TypeError):
            value = getattr(struct, "value", None)
            if isinstance(value, list):
                leaves.extend(value)
            else:
                # keep existing debug string to preserve behavior
                print(f"leave {value} is not  added")
        except:
            pass

    return leaves

def struct_to_numpy(struct: Any) -> np.ndarray:
    """Convert filtered struct into a 1D numpy array of floats."""
    return np.array(flatten_leaves(struct), dtype=float)

# ---------- Descriptor plan (compiled once) ----------

# ---------- Descriptor plan (compiled once) ----------

@dataclass(frozen=True)
class _Plan:
    # flat (top-level) keys
    rule_map: Dict[str, Rule]
    args_map: Dict[str, Dict[str, Any]]
    copied_flag: Dict[str, bool]
    # dot-path keys
    path_rule_map: Dict[Tuple[str, ...], Rule]           # ('res','action') -> rule
    path_args_map: Dict[Tuple[str, ...], Dict[str, Any]]  # ('res','action') -> kwargs
    path_copied_flag: Dict[Tuple[str, ...], bool]         # ('res','action') -> is_copied
    # copied groups seen anywhere
    copied_groups: Set[str]
    # top-level ordering (unchanged)
    order: List[str]
    order_index: Dict[str, int]


def _compile_plan(descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Optional[_Plan]:
    if descriptor is None or not isinstance(descriptor, Mapping):
        return None

    rule_map: Dict[str, Rule] = {}
    args_map: Dict[str, Dict[str, Any]] = {}
    copied_flag: Dict[str, bool] = {}

    path_rule_map: Dict[Tuple[str, ...], Rule] = {}
    path_args_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    path_copied_flag: Dict[Tuple[str, ...], bool] = {}

    copied_groups: Set[str] = set()
    order: List[str] = list(descriptor.keys())

    for k, v in descriptor.items():
        is_mapping = isinstance(v, Mapping)
        rule = v.get("rule") if is_mapping and "rule" in v else (v if not is_mapping else None)
        args = (dict(v["args"]) if is_mapping and isinstance(v.get("args"), Mapping) else None)
        copied = bool(is_mapping and "copied" in v)
        grp = v.get("copied") if is_mapping else None
        if isinstance(grp, str) and grp:
            copied_groups.add(grp)

        if "." in k:
            path = tuple(k.split("."))
            if rule is not None:
                path_rule_map[path] = rule  # type: ignore[assignment]
                if args:
                    path_args_map[path] = args
                path_copied_flag[path] = copied
        else:
            if rule is not None:
                rule_map[k] = rule  # type: ignore[assignment]
                if args:
                    args_map[k] = args
                copied_flag[k] = copied

    order_index = {k: i for i, k in enumerate(order)}
    return _Plan(
        rule_map=rule_map,
        args_map=args_map,
        copied_flag=copied_flag,
        path_rule_map=path_rule_map,
        path_args_map=path_args_map,
        path_copied_flag=path_copied_flag,
        copied_groups=copied_groups,
        order=order,
        order_index=order_index,
    )

# ---------- Core filters (fast) ----------

def _apply_rule_fast(value: Any, rule: Rule, args: Optional[dict]) -> Any:
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
        out = rule(value, **(args or {}))
        if out is None or (isinstance(out, dict) and not out):
            return None
        return out
    if isinstance(rule, Mapping):
        # handled by recursive filter
        return None
    return None

def _get_by_path(d: Any, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    cur = d
    for seg in path:
        if not isinstance(cur, dict) or seg not in cur:
            return False, None
        cur = cur[seg]
    return True, cur

def _set_by_path(dst: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = dst
    for seg in path[:-1]:
        nxt = cur.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[seg] = nxt
        cur = nxt
    cur[path[-1]] = value

def _filter_mapping_rec(v: Any, rule_mapping: Mapping[str, Any]) -> Any:
    """Apply nested descriptor (rule is a mapping) onto dict v."""
    if not isinstance(v, dict):
        return None
    # Compile a tiny plan for this nested mapping (re-using fast path)
    nested_plan = _compile_plan(rule_mapping)  # small; ok to recompile here
    return _trace_filter_fast(v, nested_plan, descriptor=rule_mapping, in_copied=False)

def _trace_filter_fast(
    trace: Any,
    plan: Optional[_Plan],
    *,
    descriptor: Optional[Descriptor | Mapping[str, Any]],
    in_copied: bool = False
) -> Any:
    if plan is None or descriptor is None or not isinstance(trace, dict):
        return trace

    rule_map = plan.rule_map
    args_map = plan.args_map
    copied_flag = plan.copied_flag

    out: Dict[str, Any] = {}

    # 1) Handle simple top-level keys in descriptor order
    for key in plan.order:
        if key not in rule_map:
            continue
        if key not in trace:
            continue
        if copied_flag.get(key, False) and not in_copied:
            continue

        v = trace[key]
        rule = rule_map[key]
        args = args_map.get(key)

        if isinstance(rule, Mapping):
            if isinstance(v, dict):
                child = _trace_filter_fast(v, _compile_plan(rule), descriptor=rule, in_copied=in_copied)
                if isinstance(child, dict) and child:
                    out[key] = child
        else:
            filtered_v = _apply_rule_fast(v, rule, args)
            if filtered_v is not None and (not isinstance(filtered_v, dict) or filtered_v):
                out[key] = filtered_v

    # 2) Handle dot-path keys (exact cascaded matches only)
    for path, rule in plan.path_rule_map.items():
        # respect copied gating for this *path* entry
        if plan.path_copied_flag.get(path, False) and not in_copied:
            continue

        found, leaf = _get_by_path(trace, path)
        if not found:
            continue

        args = plan.path_args_map.get(path)
        if isinstance(rule, Mapping):
            filtered_leaf = _filter_mapping_rec(leaf, rule)
        else:
            filtered_leaf = _apply_rule_fast(leaf, rule, args)

        if filtered_leaf is None or (isinstance(filtered_leaf, dict) and not filtered_leaf):
            continue

        # write back only this exact path into output
        _set_by_path(out, path, filtered_leaf)

    # 3) Recurse into unmatched children to discover nested descriptor mappings or copied groups
    desc_map = descriptor if isinstance(descriptor, Mapping) else {}
    for key, v in trace.items():
        # skip if we already produced key (either via top-level or via some dot-path root)
        if key in out:
            # if we set a nested value like res.action, 'res' is present in out already — skip
            continue

        # copied-group context propagation (unchanged)
        child_in_copied = in_copied or (key in plan.copied_groups)

        # If descriptor contains a nested mapping at this key, recurse into it
        rule_entry = desc_map.get(key)
        if isinstance(rule_entry, Mapping) and "rule" in rule_entry and isinstance(rule_entry["rule"], Mapping):
            child = _trace_filter_fast(v, _compile_plan(rule_entry["rule"]), descriptor=rule_entry["rule"], in_copied=child_in_copied)
            if isinstance(child, dict) and child:
                out[key] = child
            continue

        # Otherwise, keep walking to find deeper dot-path targets (handled by step 2 when found)
        if isinstance(v, dict):
            child = _trace_filter_fast(v, plan, descriptor=descriptor, in_copied=child_in_copied)
            if isinstance(child, dict) and child:
                out[key] = child

    return out

# ---------- Copied-view builder (supports dotted keys) ----------

def _copy_new_dict_fast2(
    trace: Dict[str, Any],
    plan: Optional[_Plan],
    descriptor: Optional[Descriptor | Mapping[str, Any]]
) -> Dict[str, Any]:
    if plan is None or descriptor is None or not isinstance(trace, dict):
        return {}

    # Build name/group index for both flat and dotted keys
    key_to_group_flat: Dict[str, str] = {}
    path_to_group: Dict[Tuple[str, ...], str] = {}
    groups: Set[str] = set()

    for k, v in descriptor.items():
        if isinstance(v, Mapping) and "copied" in v:
            grp = v.get("copied")
            if isinstance(grp, str) and grp:
                groups.add(grp)
                if "." in k:
                    path_to_group[tuple(k.split("."))] = grp
                else:
                    key_to_group_flat[k] = grp

    if not groups:
        return {}

    result: Dict[str, Any] = {grp: {} for grp in groups}

    # Helper to insert (possibly nested) into a group's bucket
    def _insert_to_group(grp: str, path: Tuple[str, ...], value: Any):
        bucket = result[grp]
        # If the descriptor had a dotted key, mirror its path structure inside the group
        _set_by_path(bucket, path, value)

    # 1) Handle dotted keys: copy exact paths if present
    for path, grp in path_to_group.items():
        found, leaf = _get_by_path(trace, path)
        if found:
            _insert_to_group(grp, path, leaf)

    # 2) Handle flat keys: copy any occurrence by name anywhere (original behavior)
    def walk(node: Any, parents: Tuple[str, ...] = ()):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            grp = key_to_group_flat.get(k)
            if grp is not None:
                # For flat copied keys, put them under their immediate parent chain
                _insert_to_group(grp, parents + (k,), v)
            if isinstance(v, dict):
                walk(v, parents + (k,))

    walk(trace, ())
    return result


# ---------- Ordering ----------

def _sort_top_level_keys_fast(data: Dict[str, Any], plan: Optional[_Plan]) -> Dict[str, Any]:
    """Sort only the top level using the precomputed order; leave nested dicts as-is."""
    if plan is None or not isinstance(data, dict):
        return data
    order_set = set(plan.order)
    # Keep descriptor-ordered keys first, then any extras in original order
    sorted_data = {}
    for k in plan.order:
        if k in data:
            sorted_data[k] = data[k]
    for k, v in data.items():
        if k not in order_set:
            sorted_data[k] = v
    return sorted_data

# ---------- Public API (unchanged signatures) ----------

def trace_filter(trace: Any, descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Any:
    plan = _compile_plan(descriptor)

    # Build copied view, then filter it (copied-keys only live there)
    copied_tree = _copy_new_dict_fast2(trace, plan, descriptor)
    copied_trace = _trace_filter_fast(copied_tree, plan, descriptor=descriptor, in_copied=True)

    # Build normal filtered view (copied-keys suppressed)
    filtered_trace = _trace_filter_fast(trace, plan, descriptor=descriptor, in_copied=False)

    # Merge and apply top-level stable order once
    merged = {**filtered_trace, **copied_trace}
    return _sort_top_level_keys_fast(merged, plan)

def shift_res_action_in_states(
    states: List[Dict[str, Any]],
    path: Tuple[str, ...] = ("rnn", "res", "action"),
) -> List[Dict[str, Any]]:
    """
    Shift the vector at `path` forward by one across `states`:
      - states[0][...path] becomes zeros (same length as first found action)
      - Each subsequent state's value becomes the previous state's
      - The original last value is dropped (by overwriting with previous)
    Only uses (and only edits) the filtered `states`.
    """

    def _get_by_path(d: Dict[str, Any], p: Tuple[str, ...]) -> Optional[Any]:
        cur: Any = d
        for seg in p:
            if not isinstance(cur, dict) or seg not in cur:
                return None
            cur = cur[seg]
        return cur

    def _set_by_path(d: Dict[str, Any], p: Tuple[str, ...], value: Any) -> bool:
        cur: Any = d
        for seg in p[:-1]:
            if not isinstance(cur, dict) or seg not in cur or not isinstance(cur[seg], dict):
                return False  # don't create new keys; only edit existing filtered structure
            cur = cur[seg]
        if not isinstance(cur, dict) or p[-1] not in cur:
            return False
        cur[p[-1]] = value
        return True

    # infer action length from the first state that already has the path
    n: Optional[int] = None
    for s in states:
        a = _get_by_path(s, path)
        if isinstance(a, (list, tuple)) and len(a) > 0:
            n = len(a)
            break
    if not n:
        return states  # nothing to do

    prev = [0.0] * n
    for s in states:
        cur = _get_by_path(s, path)
        # write shifted value only if the full path exists
        wrote = _set_by_path(s, path, list(prev))
        if wrote and isinstance(cur, (list, tuple)) and len(cur) == n:
            prev = list(cur)
    return states

# ---------- Example main remains identical ----------

def trace_collec(
    json_file: str,
    state_descriptor: Optional[Dict[str, Rule]] = None,
    reward_descriptor: Optional[Dict[str, Rule]] = None
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:

    with open(json_file, "r") as f:
        lines = f.readlines()

    trace_items = [json.loads(line, object_hook=_json_object_hook) for line in lines]

    actions = [t.get("action") for t in trace_items]
    states  = [trace_filter(t, state_descriptor)  for t in trace_items]
    rewards = [trace_filter(t, reward_descriptor) for t in trace_items]
    network_output = [t.get("res") for t in trace_items]

    # ---- simple shift of res.action (+1) inside states ----
    # find action length from raw traces first (fallback to states if needed)
    states = shift_res_action_in_states(states, path=("rnn", "res", "action"))

    return states, actions, rewards, network_output


if __name__ == "__main__":
    STATE_DESCRIPTOR = {
        "signal_dbm": {"rule": True, "pos": "agent"},
        "tx_mbit_s":  {"rule": True, "pos": "agent"},
        "bitrate":    {"rule": True, "pos": "agent"},
        "app_buff":   {"rule": True, "pos": "agent"},
        "frame_count":{"rule": True, "pos": "agent"},
        "queues":     {"rule": "queues_only_ac1", "pos": "agent"},
        "rtt":        {"rule": True, "pos": "flow", "copied": "rnn"},
        "outage_rate":{"rule": True, "pos": "flow", "copied": "rnn"},
        "res.action":     {"rule": True, "pos": "flow", "copied": "rnn"},
    }

    example_js_str = '''
    {"t": 1760702113.036341, "iteration": 0, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [16.354024410247803, 40.01936316490173, 498.2471615076065, 107.10836946964264]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 11}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0048943062623341875, "outage_rate": 0.004726814726988474, "throughput": 7.396829444090822, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760702113, "nanos_since_epoch": 30374143}, "queues": {"192.168.3.35": {"1": 3, "0": 0, "2": 0}, "192.168.3.25": {"0": 0, "1": 0, "2": 0}}, "link": {"192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}, "192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -57, "tx_mbit_s": 174.0}}}}, "timed_out": false, "res": {"action": [-0.9345839023590088, -0.8399225473403931, 0.992988646030426, -0.5715665221214294, 0.7499494552612305], "log_prob": [-18.429311752319336], "value": 0}, "policy": "SAC"}
    '''
    example_js = trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR)
    
    
    print(example_js)
    print(flatten_leaves(example_js))
    
    
    example_js_str_2 = '''
    {"t": 1760690575.639269, "iteration": 2, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [441.0734474658966, 57.915568351745605, 17.910495400428772, 477.911114692688]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 12}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.004517078399658203, "outage_rate": 0.0, "throughput": 7.753064995008742, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760690575, "nanos_since_epoch": 638260169}, "queues": {"192.168.3.35": {"1": 0, "0": 0, "2": 0}, "192.168.3.25": {"2": 0, "0": 0, "1": 0}}, "link": {"192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -58, "tx_mbit_s": 174.0}, "192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}}}}, "timed_out": false, "res": {"action": [0.7642937898635864, -0.7683377265930176, -0.9283580183982849, 0.911644458770752, 0.8903278708457947], "log_prob": [-16.09775161743164], "value": 0}, "policy": "SAC"}
    '''
    states = [ trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR),  trace_filter(json.loads(example_js_str_2.replace("'", '"')), STATE_DESCRIPTOR)]
    
    import copy
    
    test_state = copy.deepcopy(states[1])
    test = shift_res_action_in_states(states, path=("rnn", "res", "action"))[1]
    
    print(test)
    print(test_state)
    print(states[1])
    