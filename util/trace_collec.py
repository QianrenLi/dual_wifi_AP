import json
import numpy as np
from typing import Any, Callable, Dict, Optional, Union, Iterable, Tuple, List
from util.control_cmd import _json_object_hook

Rule = Union[bool, Callable[[Any], Any], Dict[str, "Rule"]]

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
    """Apply a rule to a value:
       - True: include value as-is
       - callable: transform value
       - dict: treat value as dict and apply nested rules
    """
    if rule is True:
        return value
    if callable(rule):
        return rule(value)
    if isinstance(rule, dict) and isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k in rule:
                out_k = _apply_rule(v, rule[k])
                # keep only non-empty results
                if out_k is not None and (not isinstance(out_k, dict) or len(out_k) > 0):
                    out[k] = out_k
        return out
    # Rule doesn't match the value type → drop
    return None

def trace_filter(trace: Any, descriptor: Optional[Dict[str, Rule]]) -> Any:
    """Filter an arbitrarily nested dict 'trace' using 'descriptor' rules.
       If descriptor is None, return trace unchanged.
       Returns a filtered dict (possibly empty) or a transformed value.
    """
    if descriptor is None:
        return trace
    if not isinstance(trace, dict):
        return trace

    filtered: Dict[str, Any] = {}

    # First: direct hits (keys present in descriptor)
    for key, rule in descriptor.items():
        if key in trace:
            v = trace[key]
            filtered_v = _apply_rule(v, rule)
            if filtered_v is not None and (not isinstance(filtered_v, dict) or len(filtered_v) > 0):
                filtered[key] = filtered_v

    # Second: keep walking into children to discover matches deeper
    for key, v in trace.items():
        if key in descriptor:
            continue  # already handled
        if isinstance(v, dict):
            child = trace_filter(v, descriptor)
            if isinstance(child, dict) and len(child) > 0:
                # Only include if something inside matched
                filtered[key] = child

    return filtered

# ---------- Example custom rules ----------
def queues_only_ac1(queues: Any, ac_key: Union[str, int] = 1, default: int = 0) -> Dict[str, Dict[str, int]]:
    """
    queues value looks like:
      {
        "192.168.3.61": {"2": 0, "0": 0},
        "192.168.3.25": {"0": 0, "2": 0}
      }
    We return:
      {
        "192.168.3.61": {"1": 0},
        "192.168.3.25": {"1": 0}
      }
    (use default if AC '1' missing)
    """
    if not isinstance(queues, dict):
        return {}
    want_keys = {str(ac_key), int(ac_key) if isinstance(ac_key, str) and ac_key.isdigit() else ac_key}
    out: Dict[str, Dict[str, int]] = {}
    for dev, ac_map in queues.items():
        if not isinstance(ac_map, dict):
            continue
        # accept both str and int forms
        val = None
        for k in want_keys:
            if k in ac_map:
                val = ac_map[k]
                break
        if val is None:
            val = default
        out[str(dev)] = {str(ac_key): int(val) if isinstance(val, (int, float)) else default}
    return out

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
    import sys
    if len(sys.argv) != 2:
        print("Usage: python trace_collec.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    # Build descriptors with rules
    STATE_DESCRIPTOR: Dict[str, Rule] = {
        # keep these fields as-is wherever they appear
        "signal_dbm": True,
        "tx_mbit_s": True,
        "bitrate": True,
        "app_buff": True,
        "frame_count": True,
        # custom rule: only AC 1, default 0 if missing
        "queues": queues_only_ac1,
    }

    REWARD_DESCRIPTOR: Dict[str, Rule] = {
        "rtt": True,
        "bitrate": True,
        "outage_rate": True,
    }

    states, actions, rewards = trace_collec(json_file, STATE_DESCRIPTOR, REWARD_DESCRIPTOR)
    print(states)
    print(actions)
    print(rewards)
