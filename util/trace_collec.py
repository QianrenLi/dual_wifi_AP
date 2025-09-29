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
            # pass  # ignore non-numeric leaves
            value = struct.value
            if isinstance(value, list):
                leaves.extend(value)
            # elif isinstance(value, float):
            #     leaves.append(value)
            else:
                print(f"leave {value} is not  added")
        except:
            pass

    return leaves

def struct_to_numpy(struct: Any) -> np.ndarray:
    """Convert filtered struct into a 1D numpy array of floats."""
    return np.array(flatten_leaves(struct), dtype=float)


def _apply_rule(value: Any, rule: Rule, args: Optional[dict] = None) -> Any:
    """Apply a single rule to a value.
       Returns transformed value, or None to indicate 'drop'.
    """
    # bool handling
    if rule is True:
        return value
    if rule is False or rule is None:
        return None

    # string -> look up registered filter and call with args
    if isinstance(rule, str):
        assert rule in FILTER_REGISTRY, f"Unknown filter '{rule}' not in FILTER_REGISTRY"
        func = FILTER_REGISTRY[rule]
        return func(value, **(args or {}))

    # callable handling
    if callable(rule):
        out = rule(value, **(args or {}))
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
            return _trace_filter(value, rule)  # type: ignore[arg-type]
        # If value isn't a dict, nothing to do
        return None

    # Unknown rule type → keep conservative and drop
    return None

def _trace_filter(trace: Any, descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Any:
    """Filter an arbitrarily nested dict 'trace' using 'descriptor' rules.
       This version also handles copying entries based on the 'copied' field."""
    if descriptor is None:
        return trace
    if not isinstance(trace, dict):
        return trace

    # Normalize descriptor into key -> rule mapping (strip 'pos')
    normalized: Dict[str, Rule] = {}
    args_map: Dict[str, Dict[str, Any]] = {}
    for k, v in descriptor.items():
        if isinstance(v, Mapping) and "rule" in v:
            normalized[k] = v["rule"]  # standard entry
            if isinstance(v.get("args"), Mapping):
                args_map[k] = dict(v["args"])
        else:
            normalized[k] = v  # type: ignore[assignment]

    filtered: Dict[str, Any] = {}

    # First pass: direct hits based on the order of the keys in descriptor
    for key in descriptor.keys():  # Ensure the keys are ordered as in descriptor
        if key in normalized and key in trace:
            v = trace[key]
            entry_args = args_map.get(key)
            filtered_v = _apply_rule(v, normalized[key], args=entry_args)
            if filtered_v is not None and (not isinstance(filtered_v, dict) or len(filtered_v) > 0):
                filtered[key] = filtered_v

    # Second pass: walk into secondary keys (nested dictionaries)
    for key, v in trace.items():
        if key in normalized:
            continue  # already handled
        if isinstance(v, dict):
            child = _trace_filter(v, descriptor)  # keep using the same normalized rules
            if isinstance(child, dict) and len(child) > 0:
                filtered[key] = child
                
    return sort_top_level_keys(filtered, descriptor)


def copy_new_dict(filtered: Dict[str, Any], descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Dict[str, Any]:
    """Create a new dictionary where copied fields (specified in the descriptor) are moved to the top level."""
    
    # A new dictionary to store the result, including copied entries at the top level
    copied_entries = {}

    def copy_recursive(trace: Dict[str, Any], parent_key: Optional[str] = None, is_root = False) -> None:
        """Recursively check for copied fields and move them to the top level, preserving the parent key."""
        for key, value in trace.items():
            # If the key has a "copied" field in the descriptor, copy the data to the top level
            if key in descriptor and "copied" in descriptor[key]:
                copied_key = descriptor[key]["copied"]
                
                # Initialize the copied entry if it doesn't exist
                if copied_key not in copied_entries:
                    copied_entries[copied_key] = {}

                # If we have a parent_key (e.g., '6203@128'), include it in the copied structure
                if parent_key:
                    if copied_key not in copied_entries:
                        copied_entries[copied_key] = {}
                    if parent_key not in copied_entries[copied_key]:
                        copied_entries[copied_key][parent_key] = {}

                    # Add the key-value pair under the parent_key in the copied structure
                    copied_entries[copied_key][parent_key][key] = value
                else:
                    # For the top-level items, just copy them as is
                    copied_entries[copied_key][key] = value

            # If the value is a nested dictionary, recurse into it
            if isinstance(value, dict):
                if not is_root:
                    copy_recursive(value, parent_key or key)
                else:
                    copy_recursive(value)

    # Call the recursive function to handle nested structures and copy fields
    copy_recursive(filtered, is_root=True)

    # Now merge the copied_entries with the filtered dictionary at the top level
    result = {**filtered, **copied_entries}

    return result


def trace_filter(trace: Any, descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Any:
    filtered_trace = _trace_filter(trace, descriptor)
    copied_trace = copy_new_dict(filtered_trace, descriptor)
    return copied_trace
    

def sort_top_level_keys(data: Dict[str, Any], descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """Sort the top-level keys according to descriptor order.
       Secondary keys (nested dictionaries) will not be sorted but are kept as they are.
    """
    if isinstance(data, dict):
        # Sort the top-level keys in the order defined in descriptor, but ensure we retain any non-empty secondary keys
        sorted_data = {}
        
        # First, process all the keys in the descriptor order
        for key in descriptor.keys():
            if key in data:
                sorted_data[key] = data[key]
        
        # Now, recursively sort nested dictionaries (secondary keys)
        for key, value in sorted_data.items():
            if isinstance(value, dict):
                sorted_data[key] = sort_top_level_keys(value, descriptor)  # Sort nested keys recursively
        
        # Now, handle any secondary keys that aren't in the descriptor at the top level
        # They should remain in their original order and not be dropped
        for key, value in data.items():
            if key not in sorted_data:
                sorted_data[key] = value

        return sorted_data
    return data


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
    network_output = [t.get("res") for t in trace_items]

    return states, actions, rewards, network_output


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
            "copied": 'rnn',
        },
        "rtt": {
            "rule": True,
            "pos": "flow",
            "copied": 'rnn',
        },
        "outage_rate": {
            "rule": True,
            "pos": "flow",
        },
    }
    
    example_js_str = '''
    {'flow_stat': {'6203@128': {'rtt': 0.0, 'outage_rate': 0.0, 'throughput': 0.0, 'throttle': 0.0, 'version': 0, 'bitrate': 2000000, 'app_buff': 0, 'frame_count': 0}}, 'device_stat': {'taken_at': {'secs_since_epoch': 1758433731, 'nanos_since_epoch': 703233756}, 'queues': {'192.168.3.61': {'0': 0, '2': 0}, '192.168.3.25': {'0': 0, '1': 3, '2': 0}}, 'link': {'192.168.3.25': {'bssid': '82:19:55:0e:6f:4e', 'ssid': 'HUAWEI-Dual-AP', 'freq_mhz': 2462, 'signal_dbm': -56, 'tx_mbit_s': 174.0}, '192.168.3.61': {'bssid': '82:19:55:0e:6f:52', 'ssid': 'HUAWEI-Dual-AP_5G', 'freq_mhz': 5745, 'tx_mbit_s': 867.0, 'signal_dbm': -48}}}}
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
