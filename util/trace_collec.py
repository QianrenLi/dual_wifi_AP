import json
import numpy as np
from typing import Any, Callable, Dict, Optional, Union, Mapping, Tuple, List
from util.control_cmd import _json_object_hook
from util.filter_rule import FILTER_REGISTRY

Rule = Union[bool, Callable[[Any], Any], Dict[str, "Rule"]]
DescriptorEntry = Dict[str, Any]  # expects {"rule": RuleType, "pos": ... (ignored)}
Descriptor = Dict[str, DescriptorEntry]
# ---------- Helper ----------

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
            pass  # ignore non-numeric leaves

    return leaves

def struct_to_numpy(struct: Any) -> np.ndarray:
    """Convert filtered struct into a 1D numpy array of floats."""
    return np.array(flatten_leaves(struct), dtype=float)


def _apply_rule(value: Any, rule: Rule) -> Any:
    """Apply a single rule to a value.
       Returns transformed value, or None to indicate 'drop'.
    """
    # bool handling
    if rule is True:
        return value
    if rule is False or rule is None:
        return None

    if isinstance(rule, str):
        assert rule in FILTER_REGISTRY, f"Unknown filter '{rule}' not in FILTER_REGISTRY"
        return _apply_rule(value, FILTER_REGISTRY[rule])

    # callable handling
    if callable(rule):
        out = rule(value)
        # Treat None / empty dict as 'dropped'
        if out is None:
            return None
        if isinstance(out, dict) and len(out) == 0:
            return None
        return out

    # nested-descriptor handling (mapping)
    if isinstance(rule, Mapping):
        # Interpret this as a nested descriptor. Only makes sense if value is a dict.
        if isinstance(value, dict):
            return trace_filter(value, rule)  # type: ignore[arg-type]
        # If value isn't a dict, nothing to do
        return None

    # Unknown rule type → keep conservative and drop
    return None

def trace_filter(trace: Any, descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Any:
    """Filter an arbitrarily nested dict 'trace' using 'descriptor' rules.
       Only 'rule' is honored; 'pos' is ignored.
       If descriptor is None, return trace unchanged.
       Returns a filtered dict (possibly empty) or a transformed value.
    """
    if descriptor is None:
        return trace
    if not isinstance(trace, dict):
        return trace

    # Normalize descriptor into key -> rule mapping (strip 'pos')
    # Allow passing a raw nested rule-mapping (when recursing)
    normalized: Dict[str, Rule] = {}
    for k, v in descriptor.items():
        if isinstance(v, Mapping) and "rule" in v:
            normalized[k] = v["rule"]  # standard entry
        else:
            # If caller passed a raw mapping of key->RuleType (nested case), accept it directly
            normalized[k] = v  # type: ignore[assignment]

    filtered: Dict[str, Any] = {}

    # First pass: direct hits
    for key, rule in normalized.items():
        if key in trace:
            v = trace[key]
            filtered_v = _apply_rule(v, rule)
            if filtered_v is not None and (not isinstance(filtered_v, dict) or len(filtered_v) > 0):
                filtered[key] = filtered_v

    # Second pass: walk into children not directly matched to discover deeper matches
    for key, v in trace.items():
        if key in normalized:
            continue  # already handled
        if isinstance(v, dict):
            child = trace_filter(v, normalized)  # keep using the same normalized rules
            if isinstance(child, dict) and len(child) > 0:
                filtered[key] = child

    return filtered

# ---------- Main function ----------
def trace_collec(
    json_file: str,
    state_descriptor: Optional[Dict[str, Rule]] = None,
    reward_descriptor: Optional[Dict[str, Rule]] = None
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]]]:

    with open(json_file, "r") as f:
        lines = f.readlines()

    trace_items = [json.loads(line, object_hook=_json_object_hook) for line in lines]

    actions = [t.get("action") for t in trace_items]
    states  = [trace_filter(t, state_descriptor)  for t in trace_items]
    rewards = [trace_filter(t, reward_descriptor) for t in trace_items]

    return states, actions, rewards


if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python trace_collec.py <json_file>")
    #     sys.exit(1)

    # json_file = sys.argv[1]

    # # Build descriptors with rules
    STATE_DESCRIPTOR = {
        "signal_dbm": {
            "rule": True,
            "pos": "agent",
        },
        "tx_mbit_s": {
            "rule": True,
            "pos": "agent",
        },
        "bitrate": {
            "rule": True,
            "pos": "agent",
        },
        "app_buff": {
            "rule": True,
            "pos": "agent",
        },
        "frame_count": {
            "rule": True,
            "pos": "agent",
        },
        "queues": {
            "rule": "queues_only_ac1",
            "pos": "agent",
        },
    }
    
    example_js_str = '''
    {'flow_stat': {'6203@128': {'rtt': 0.0, 'outage_rate': 0.0, 'throughput': 0.0, 'throttle': 0.0, 'version': 0, 'bitrate': 2000000, 'app_buff': 0, 'frame_count': 0}}, 'device_stat': {'taken_at': {'secs_since_epoch': 1758433731, 'nanos_since_epoch': 703233756}, 'queues': {'192.168.3.61': {'0': 0, '2': 0}, '192.168.3.25': {'0': 0, '1': 3, '2': 0}}, 'link': {'192.168.3.25': {'bssid': '82:19:55:0e:6f:4e', 'ssid': 'HUAWEI-Dual-AP', 'freq_mhz': 2462, 'signal_dbm': -56, 'tx_mbit_s': 174.0}, '192.168.3.61': {'bssid': '82:19:55:0e:6f:52', 'ssid': 'HUAWEI-Dual-AP_5G', 'freq_mhz': 5745, 'signal_dbm': -48, 'tx_mbit_s': 867.0}}}}
    '''

    example_js = trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR)
    print(example_js)
    
    # REWARD_DESCRIPTOR: Dict[str, Rule] = {
    #     "rtt": True,
    #     "bitrate": True,
    #     "outage_rate": True,
    # }

    # states, actions, rewards = trace_collec(json_file, STATE_DESCRIPTOR, REWARD_DESCRIPTOR)
    # print(states)
    # print(actions)
    # print(rewards)
