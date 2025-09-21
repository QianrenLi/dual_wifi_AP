from . import register_filter 

@register_filter
def queues_only_ac1(queues, ac_key = 1, default: int = 0):
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
    out = {}
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