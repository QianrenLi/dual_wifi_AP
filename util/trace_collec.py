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

@dataclass(frozen=True)
class _Plan:
    rule_map: Dict[str, Rule]          # key -> rule
    args_map: Dict[str, Dict[str, Any]]# key -> kwargs for rule
    copied_flag: Dict[str, bool]       # key -> is_copied
    copied_groups: Set[str]            # e.g. {'rnn'}
    order: List[str]                   # descriptor key order (for stable top-level layout)
    order_index: Dict[str, int]        # key -> rank for O(1) ordering

def _compile_plan(descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Optional[_Plan]:
    if descriptor is None:
        return None
    if not isinstance(descriptor, Mapping):
        return None

    rule_map: Dict[str, Rule] = {}
    args_map: Dict[str, Dict[str, Any]] = {}
    copied_flag: Dict[str, bool] = {}
    copied_groups: Set[str] = set()
    order: List[str] = list(descriptor.keys())

    for k, v in descriptor.items():
        if isinstance(v, Mapping) and "rule" in v:
            rule_map[k] = v["rule"]
            copied = "copied" in v
            copied_flag[k] = copied
            if copied:
                grp = v.get("copied")
                if isinstance(grp, str) and grp:
                    copied_groups.add(grp)
            if isinstance(v.get("args"), Mapping):
                args_map[k] = dict(v["args"])
        else:
            rule_map[k] = v  # type: ignore[assignment]
            copied_flag[k] = False

    order_index = {k: i for i, k in enumerate(order)}
    return _Plan(rule_map, args_map, copied_flag, copied_groups, order, order_index)

# ---------- Core filters (fast) ----------

def _apply_rule_fast(value: Any, rule: Rule, args: Optional[dict]) -> Any:
    """Apply a rule to a value with minimal overhead. Return None to drop."""
    # bool handling
    if rule is True:
        return value
    if rule is False or rule is None:
        return None

    # registry filter name
    if isinstance(rule, str):
        func = FILTER_REGISTRY.get(rule)
        if func is None:
            raise AssertionError(f"Unknown filter '{rule}' not in FILTER_REGISTRY")
        return func(value, **(args or {}))

    # callable rule
    if callable(rule):
        out = rule(value, **(args or {}))
        if out is None:
            return None
        if isinstance(out, dict) and not out:
            return None
        return out

    # nested descriptor form (mapping) is handled by _trace_filter_fast
    # but if it's reached here and value is not a dict, drop:
    if isinstance(rule, Mapping):
        return None

    # unknown type -> drop
    return None

def _trace_filter_fast(
    trace: Any,
    plan: Optional[_Plan],
    *,
    descriptor: Optional[Descriptor | Mapping[str, Any]],
    in_copied: bool = False
) -> Any:
    """Filter nested dict using a compiled plan. Suppress copied-keys outside copied groups."""
    if plan is None or descriptor is None:
        return trace
    if not isinstance(trace, dict):
        return trace

    rule_map = plan.rule_map
    args_map = plan.args_map
    copied_flag = plan.copied_flag
    copied_groups = plan.copied_groups

    out: Dict[str, Any] = {}

    # First: process keys present in the descriptor order
    for key in plan.order:
        if key not in rule_map:
            continue
        if key not in trace:
            continue
        # If this key is marked 'copied', include only when in_copied
        if copied_flag.get(key, False) and not in_copied:
            continue
        v = trace[key]
        rule = rule_map[key]
        args = args_map.get(key)

        # Nested descriptor case: rule is a mapping → recurse on dict values
        if isinstance(rule, Mapping):
            if isinstance(v, dict):
                child = _trace_filter_fast(v, plan, descriptor=rule, in_copied=in_copied)
                if isinstance(child, dict) and child:
                    out[key] = child
            # else drop
        else:
            filtered_v = _apply_rule_fast(v, rule, args)
            if filtered_v is not None and (not isinstance(filtered_v, dict) or filtered_v):
                out[key] = filtered_v

    # Second: walk other dict children not explicitly listed in descriptor
    for key, v in trace.items():
        if key in rule_map:
            continue
        if isinstance(v, dict):
            # Flip context if we step into a copied-group container (e.g., 'rnn')
            child_in_copied = in_copied or (key in copied_groups)
            child = _trace_filter_fast(v, plan, descriptor=descriptor, in_copied=child_in_copied)
            if isinstance(child, dict) and child:
                out[key] = child

    return out

def _copy_new_dict_fast(trace: Dict[str, Any], plan: Optional[_Plan]) -> Dict[str, Any]:
    """Build structure hosting only the copied fields under their group names."""
    if plan is None or not isinstance(trace, dict):
        return {}

    # Build key->group map and group containers
    key_to_group: Dict[str, str] = {}
    groups: Set[str] = set()
    for k, is_copied in plan.copied_flag.items():
        if is_copied:
            # The original descriptor entry holds group name; we’ll re-derive from rule_map lookup:
            # Locate group name by scanning the original rule entry only once via a cached map.
            # For simplicity, we infer group by checking descriptor again (cheap; few keys).
            # To keep it O(K), we create a small one-time resolver:
            pass

    # Build a resolver to get group name for a copied key
    # We reconstruct it once from the (small) descriptor dicts referenced by plan.order
    # This keeps behavior identical to the previous version.
    group_resolver: Dict[str, str] = {}
    # NOTE: We don’t have raw descriptor here, so we pass it into this function instead of plan-only.
    # Adjust signature to accept descriptor too.
    raise NotImplementedError("Internal usage error: _copy_new_dict_fast requires descriptor.")

# Revised version with descriptor available (so we can read 'copied' names):
def _copy_new_dict_fast2(trace: Dict[str, Any], plan: Optional[_Plan], descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Dict[str, Any]:
    if plan is None or descriptor is None or not isinstance(trace, dict):
        return {}

    key_to_group: Dict[str, str] = {}
    groups: Set[str] = set()
    for k, v in descriptor.items():
        if isinstance(v, Mapping) and "copied" in v:
            grp = v.get("copied")
            if isinstance(grp, str) and grp:
                key_to_group[k] = grp
                groups.add(grp)

    if not key_to_group:
        return {}

    result: Dict[str, Any] = {grp: {} for grp in groups}

    def walk(node: Any, parent_key: Optional[str] = None):
        if not isinstance(node, dict):
            return
        # Localize to speed attribute & method lookups
        r_local = result
        ktg = key_to_group

        for k, v in node.items():
            grp = ktg.get(k)
            if grp is not None:
                bucket = r_local[grp]
                if parent_key is not None:
                    slot = bucket.get(parent_key)
                    if slot is None:
                        slot = {}
                        bucket[parent_key] = slot
                    slot[k] = v
                else:
                    # top-level
                    if k not in bucket:
                        bucket[k] = v
            if isinstance(v, dict):
                walk(v, parent_key or k)

    walk(trace, None)
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

    return states, actions, rewards, network_output



if __name__ == "__main__":
    STATE_DESCRIPTOR = {
        "signal_dbm": {"rule": True, "pos": "agent"},
        "tx_mbit_s":  {"rule": True, "pos": "agent"},
        "bitrate":    {"rule": True, "pos": "agent"},
        "app_buff":   {"rule": True, "pos": "agent"},
        "frame_count":{"rule": True, "pos": "agent"},
        "queues":     {"rule": "queues_only_ac1", "pos": "agent", "copied": "rnn"},
        "rtt":        {"rule": True, "pos": "flow", "copied": "rnn"},
        "outage_rate":{"rule": True, "pos": "flow"},
    }

    example_js_str = '''
    {'flow_stat': {'6203@128': {'rtt': 0.0, 'outage_rate': 0.0, 'throughput': 0.0, 'throttle': 0.0, 'version': 0, 'bitrate': 2000000, 'app_buff': 0, 'frame_count': 0}}, 'device_stat': {'taken_at': {'secs_since_epoch': 1758433731, 'nanos_since_epoch': 703233756}, 'queues': {'192.168.3.61': {'0': 0, '2': 0}, '192.168.3.25': {'0': 0, '1': 3, '2': 0}}, 'link': {'192.168.3.25': {'bssid': '82:19:55:0e:6f:4e', 'ssid': 'HUAWEI-Dual-AP', 'freq_mhz': 2462, 'signal_dbm': -56, 'tx_mbit_s': 174.0}, '192.168.3.61': {'bssid': '82:19:55:0e:6f:52', 'ssid': 'HUAWEI-Dual-AP_5G', 'freq_mhz': 5745, 'tx_mbit_s': 867.0, 'signal_dbm': -48}}}}
    '''
    example_js = trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR)
    print(example_js)
    print(flatten_leaves(example_js))