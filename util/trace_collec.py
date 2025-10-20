import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, Mapping, Tuple, List, Set, Iterable
from functools import lru_cache

from util.control_cmd import _json_object_hook
from util.filter_rule import FILTER_REGISTRY

# ----------------------------- Types ----------------------------- #
Rule = Union[bool, Callable[[Any], Any], Dict[str, "Rule"]]
DescriptorEntry = Dict[str, Any]
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
                print(f"leave {value} is not  added")
        except Exception:
            pass
    return out

def struct_to_numpy(struct: Any) -> np.ndarray:
    return np.array(flatten_leaves(struct), dtype=float)

# ========================= Descriptor freeze ===================== #
# 目标：为 _compile_plan 做 LRU 缓存制作“可哈希且保序”的 key

def _canon(obj: Any) -> Any:
    """把 descriptor 中不可序列化对象（尤其是 callable）转为可 JSON 的形式。
       - callable → {"__callable__": qualname or repr}
       - 其余保持结构，映射转 list[(k,v)] 以保序。"""
    if callable(obj):
        name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
        return {"__callable__": str(name)}
    if isinstance(obj, Mapping):
        # 保序关键：转成按原顺序的 (k, v) 对序列
        return [{"k": k, "v": _canon(v)} for k, v in obj.items()]
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    # 标量 / str / bool / None
    return obj

def _freeze_descriptor(descriptor: Mapping[str, Any]) -> str:
    """把 descriptor 冻结为一个 JSON 字符串（保留原键顺序），作为缓存 key。"""
    return json.dumps(_canon(descriptor), separators=(",", ":"), ensure_ascii=False)

# ========================= Plan (compiled) ======================= #

@dataclass(frozen=True)
class _Plan:
    # flat keys
    rule_map: Dict[str, Rule]
    args_map: Dict[str, Dict[str, Any]]
    copied_flag: Dict[str, bool]
    # dotted keys
    path_rule_map: Dict[Tuple[str, ...], Rule]
    path_args_map: Dict[Tuple[str, ...], Dict[str, Any]]
    path_copied_flag: Dict[Tuple[str, ...], bool]
    # copied groups
    copied_groups: Set[str]
    # stable top-level order
    order: List[str]
    order_index: Dict[str, int]
    # precompiled path accessors
    path_getters: Dict[Tuple[str, ...], Callable[[Dict[str, Any]], Tuple[bool, Any]]]
    path_setters: Dict[Tuple[str, ...], Callable[[Dict[str, Any], Any], None]]

def _make_getter(path: Tuple[str, ...]) -> Callable[[Dict[str, Any]], Tuple[bool, Any]]:
    # 局部变量绑定，避免循环内创建临时对象
    p = path
    def _get(d: Dict[str, Any]) -> Tuple[bool, Any]:
        cur: Any = d
        for seg in p:
            if not isinstance(cur, dict) or seg not in cur:
                return False, None
            cur = cur[seg]
        return True, cur
    return _get

def _make_setter(path: Tuple[str, ...]) -> Callable[[Dict[str, Any], Any], None]:
    p = path
    last = p[-1]
    pre = p[:-1]
    def _set(dst: Dict[str, Any], value: Any) -> None:
        cur = dst
        for seg in pre:
            nxt = cur.get(seg)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[seg] = nxt
            cur = nxt
        cur[last] = value
    return _set

# 供嵌套 mapping 也能复用缓存：接收“已冻结”的 JSON 字符串
@lru_cache(maxsize=256)
def _compile_plan_from_frozen(frozen: str) -> _Plan:
    # 反序列化回“保序对象”（list of {"k","v"} 形式），再还原为 Mapping
    def _thaw(x: Any) -> Any:
        if isinstance(x, list) and x and isinstance(x[0], dict) and "k" in x[0] and "v" in x[0]:
            return {item["k"]: _thaw(item["v"]) for item in x}
        if isinstance(x, list):
            return [_thaw(t) for t in x]
        if isinstance(x, dict) and "__callable__" in x:
            # 无法反向恢复原函数，但在 plan 中仅存储 rule 原对象，
            # 这里仅用于生成结构，rule 本体会由上层 _compile_plan 注入。
            return x  # 占位，不会直接使用
        if isinstance(x, dict):
            return {k: _thaw(v) for k, v in x.items()}
        return x

    # 注意：真正的 rule/callable 对象由上层 _compile_plan 注入；
    # 这里仅利用结构重建 order 等信息。
    thawed = _thaw(json.loads(frozen))

    # 构建空 Plan；字段会在 _compile_plan 中填充真实 rule
    return _Plan(
        rule_map={}, args_map={}, copied_flag={},
        path_rule_map={}, path_args_map={}, path_copied_flag={},
        copied_groups=set(),
        order=list(thawed.keys()) if isinstance(thawed, Mapping) else [],
        order_index={},
        path_getters={}, path_setters={}
    )

def _compile_plan(descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Optional[_Plan]:
    if not isinstance(descriptor, Mapping) or not descriptor:
        return None

    frozen = _freeze_descriptor(descriptor)
    # 先取结构骨架（含 order），再注入真实规则对象
    base = _compile_plan_from_frozen(frozen)

    rule_map: Dict[str, Rule] = {}
    args_map: Dict[str, Dict[str, Any]] = {}
    copied_flag: Dict[str, bool] = {}

    path_rule_map: Dict[Tuple[str, ...], Rule] = {}
    path_args_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    path_copied_flag: Dict[Tuple[str, ...], bool] = {}

    copied_groups: Set[str] = set()
    order = list(descriptor.keys())
    order_index = {k: i for i, k in enumerate(order)}

    # 预编译 dotted path 的 getter/setter
    path_getters: Dict[Tuple[str, ...], Callable[[Dict[str, Any]], Tuple[bool, Any]]] = {}
    path_setters: Dict[Tuple[str, ...], Callable[[Dict[str, Any], Any], None]] = {}

    for key, val in descriptor.items():
        is_map = isinstance(val, Mapping)
        rule: Optional[Rule] = (val.get("rule") if is_map else val) if (not is_map or "rule" in val) else None
        args: Optional[Dict[str, Any]] = (dict(val["args"]) if is_map and isinstance(val.get("args"), Mapping) else None)
        copied = bool(is_map and "copied" in val)
        grp = val.get("copied") if is_map else None
        if isinstance(grp, str) and grp:
            copied_groups.add(grp)

        if "." in key:
            path = tuple(key.split("."))
            if rule is not None:
                path_rule_map[path] = rule
                if args:
                    path_args_map[path] = args
                path_copied_flag[path] = copied
                path_getters[path] = _make_getter(path)
                path_setters[path] = _make_setter(path)
        else:
            if rule is not None:
                rule_map[key] = rule
                if args:
                    args_map[key] = args
                copied_flag[key] = copied

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
        path_getters=path_getters,
        path_setters=path_setters
    )

# ========================= Core filtering ======================== #

def _apply_rule_fast(value: Any, rule: Rule, args: Optional[dict]) -> Any:
    if rule is True:
        return value
    if rule is False or rule is None:
        return None
    if isinstance(rule, str):
        func = FILTER_REGISTRY.get(rule)
        if func is None:
            # 与原行为一致：直接异常更易暴露配置错误
            raise AssertionError(f"Unknown filter '{rule}' not in FILTER_REGISTRY")
        return func(value, **(args or {}))
    if callable(rule):
        out = rule(value, **(args or {}))
        return out if (out is not None and (not isinstance(out, dict) or out)) else None
    if isinstance(rule, Mapping):
        # 交给递归映射处理
        return None
    return None

def _filter_mapping_rec(v: Any, rule_mapping: Mapping[str, Any]) -> Any:
    if not isinstance(v, dict):
        return None
    nested_plan = _compile_plan(rule_mapping)  # 借助缓存，常量时间
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

    out: Dict[str, Any] = {}

    # 1) 顶层键：按 descriptor 顺序
    for key in plan.order:
        rule = plan.rule_map.get(key)
        if rule is None or key not in trace:
            continue
        if plan.copied_flag.get(key, False) and not in_copied:
            continue

        v = trace[key]
        if isinstance(rule, Mapping):
            if isinstance(v, dict):
                child = _trace_filter_fast(v, _compile_plan(rule), descriptor=rule, in_copied=in_copied)
                if isinstance(child, dict) and child:
                    out[key] = child
        else:
            filtered = _apply_rule_fast(v, rule, plan.args_map.get(key))
            if filtered is not None and (not isinstance(filtered, dict) or filtered):
                out[key] = filtered

    # 2) 点路径：精确匹配
    for path, rule in plan.path_rule_map.items():
        if plan.path_copied_flag.get(path, False) and not in_copied:
            continue
        ok, leaf = plan.path_getters[path](trace)
        if not ok:
            continue
        args = plan.path_args_map.get(path)
        filtered = _filter_mapping_rec(leaf, rule) if isinstance(rule, Mapping) else _apply_rule_fast(leaf, rule, args)
        if filtered is None or (isinstance(filtered, dict) and not filtered):
            continue
        plan.path_setters[path](out, filtered)

    # 3) 递归发现更深规则/传播 copied 语义
    desc_map = descriptor if isinstance(descriptor, Mapping) else {}
    for key, v in trace.items():
        if key in out:
            continue
        child_in_copied = in_copied or (key in plan.copied_groups)

        rule_entry = desc_map.get(key)
        if isinstance(rule_entry, Mapping) and isinstance(rule_entry.get("rule"), Mapping):
            child = _trace_filter_fast(v, _compile_plan(rule_entry["rule"]), descriptor=rule_entry["rule"], in_copied=child_in_copied)
            if isinstance(child, dict) and child:
                out[key] = child
        elif isinstance(v, dict):
            # 继续向下让点路径在 step 2 命中
            child = _trace_filter_fast(v, plan, descriptor=descriptor, in_copied=child_in_copied)
            if isinstance(child, dict) and child:
                out[key] = child

    return out

# ======================= Copied view builder ===================== #

def _copy_new_dict_fast2(
    trace: Dict[str, Any],
    plan: Optional[_Plan],
    descriptor: Optional[Descriptor | Mapping[str, Any]]
) -> Dict[str, Any]:
    if plan is None or descriptor is None or not isinstance(trace, dict):
        return {}

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

    # 点路径：按原路径落位
    for path, grp in path_to_group.items():
        ok, leaf = plan.path_getters.get(path, _make_getter(path))(trace)
        if ok:
            plan.path_setters.get(path, _make_setter(path))(result[grp], leaf)

    # 顶层 copied 键：任意深度遇到同名键都复制（与原实现一致）
    def _walk(node: Any, parents: Tuple[str, ...] = ()):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            grp = key_to_group_flat.get(k)
            if grp is not None:
                # 在 group 中以 parents 路径重建落位
                setter = plan.path_setters.get(parents + (k,), _make_setter(parents + (k,)))
                setter(result[grp], v)
            if isinstance(v, dict):
                _walk(v, parents + (k,))
    _walk(trace)
    return result

# ============================ Sorting ============================ #

def _sort_top_level_keys_fast(data: Dict[str, Any], plan: Optional[_Plan]) -> Dict[str, Any]:
    """仅在顶层按 descriptor 顺序稳定排序；未声明的顶层键保持原插入顺序。"""
    if plan is None or not isinstance(data, dict):
        return data
    ordered = {k: data[k] for k in plan.order if k in data}
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = v
    return ordered

def _deep_sort_mappings(node: Any) -> Any:
    """仅对（IPv4 键 / 纯数字字符串键）排序；其它保持原序。"""
    if isinstance(node, dict):
        # 先对子节点做排序，再根据 key 类型决定是否重排当前层
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

# ============================ Public API ========================= #

def trace_filter(trace: Any, descriptor: Optional[Descriptor | Mapping[str, Any]]) -> Any:
    """对单条 trace 应用描述符过滤；合并普通视图与 copied 视图；稳定排序。"""
    plan = _compile_plan(descriptor)

    copied_tree = _copy_new_dict_fast2(trace, plan, descriptor)
    copied_trace = _trace_filter_fast(copied_tree, plan, descriptor=descriptor, in_copied=True)

    filtered_trace = _trace_filter_fast(trace, plan, descriptor=descriptor, in_copied=False)

    merged = {**filtered_trace, **copied_trace}
    return _deep_sort_mappings(_sort_top_level_keys_fast(merged, plan))

def shift_res_action_in_states(
    states: List[Dict[str, Any]],
    path: Tuple[str, ...] = ("rnn", "res", "action"),
) -> List[Dict[str, Any]]:
    """把 states 中 path 指向的向量整体右移 1（首个置零），仅当路径已存在时写入。"""

    # 局部 getter/setter，避免重复创建
    get = _make_getter(path)
    # 只在写入时要求各级已存在（不创建新键）
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
        ok, a = get(s)
        if ok and isinstance(a, (list, tuple)) and a:
            n = len(a)
            break
    if not n:
        return states

    prev = [0.0] * n
    for s in states:
        ok, cur = get(s)
        wrote = _set_if_exists(s, list(prev))
        if wrote and ok and isinstance(cur, (list, tuple)) and len(cur) == n:
            prev = list(cur)
    return states

def trace_collec(
    json_file: str,
    state_descriptor: Optional[Dict[str, Rule]] = None,
    reward_descriptor: Optional[Dict[str, Rule]] = None
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(json_file, "r") as f:
        trace_items = [json.loads(line, object_hook=_json_object_hook) for line in f]

    actions = [t.get("action") for t in trace_items]
    states  = [trace_filter(t, state_descriptor)  for t in trace_items]
    rewards = [trace_filter(t, reward_descriptor) for t in trace_items]
    network_output = [t.get("res") for t in trace_items]

    # in-place shift of res.action (+1) inside states
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
        # "outage_rate":{"rule": True, "pos": "flow"},
        "res.action":     {"rule": True, "pos": "flow", "copied": "rnn"},
    }

    example_js_str = '''
    {"t": 1760702113.036341, "iteration": 0, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [16.354024410247803, 40.01936316490173, 498.2471615076065, 107.10836946964264]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 11}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0048943062623341875, "outage_rate": 0.004726814726988474, "throughput": 7.396829444090822, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760702113, "nanos_since_epoch": 30374143}, "queues": {"192.168.3.25": {"1": 3, "0": 0, "2": 0}, "192.168.3.35": {"0": 0, "1": 0, "2": 0}}, "link": {"192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}, "192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -57, "tx_mbit_s": 174.0}}}}, "timed_out": false, "policy": "SAC"}
    '''
    
    example_js_str2 = '''
    {"t": 1760702113.036341, "iteration": 0, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [16.354024410247803, 40.01936316490173, 498.2471615076065, 107.10836946964264]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 11}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.0048943062623341875, "outage_rate": 0.004726814726988474, "throughput": 7.396829444090822, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760702113, "nanos_since_epoch": 30374143}, "queues": {"192.168.3.35": {"1": 3, "0": 0, "2": 0}, "192.168.3.25": {"0": 0, "1": 0, "2": 0}}, "link": {"192.168.3.25": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}, "192.168.3.35": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -57, "tx_mbit_s": 174.0}}}}, "timed_out": false, "policy": "SAC"}
    '''
    
    example_js_str3 = '''
    {"t": 1760690575.639269, "iteration": 2, "links": ["6203@128"], "action": {"6203@128": {"__class__": "ControlCmd", "__data__": {"policy_parameters": {"__class__": "C_LIST_FLOAT_DIM4_0_500", "__data__": [441.0734474658966, 57.915568351745605, 17.910495400428772, 477.911114692688]}, "version": {"__class__": "C_INT_RANGE_0_13", "__data__": 12}}}}, "stats": {"flow_stat": {"6203@128": {"rtt": 0.004517078399658203, "outage_rate": 0.0, "throughput": 7.753064995008742, "throttle": 0.0, "bitrate": 2000000, "app_buff": 0, "frame_count": 0}}, "device_stat": {"taken_at": {"secs_since_epoch": 1760690575, "nanos_since_epoch": 638260169}, "queues": {"192.168.3.35": {"1": 0, "0": 0, "2": 0}, "192.168.3.25": {"2": 0, "0": 0, "1": 0}}, "link": {"192.168.3.25": {"bssid": "82:19:55:0e:6f:4e", "ssid": "HUAWEI-Dual-AP", "freq_mhz": 2462, "signal_dbm": -58, "tx_mbit_s": 174.0}, "192.168.3.35": {"bssid": "82:19:55:0e:6f:52", "ssid": "HUAWEI-Dual-AP_5G", "freq_mhz": 5745, "signal_dbm": -50, "tx_mbit_s": 867.0}}}}, "timed_out": false, "res": {"action": [0.7642937898635864, -0.7683377265930176, -0.9283580183982849, 0.911644458770752, 0.8903278708457947], "log_prob": [-16.09775161743164], "value": 0}, "policy": "SAC"}
    '''
    
    print(trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR))
    print(flatten_leaves(trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR)))
    print(trace_filter(json.loads(example_js_str2.replace("'", '"')), STATE_DESCRIPTOR))
    print(flatten_leaves(trace_filter(json.loads(example_js_str2.replace("'", '"')), STATE_DESCRIPTOR)))
    print(trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR))
    print(flatten_leaves(trace_filter(json.loads(example_js_str3.replace("'", '"')), STATE_DESCRIPTOR)))
    
    

    # states = [ trace_filter(json.loads(example_js_str.replace("'", '"')), STATE_DESCRIPTOR),  trace_filter(json.loads(example_js_str_2.replace("'", '"')), STATE_DESCRIPTOR)]
    
    # import copy
    
    # test_state = copy.deepcopy(states[1])
    # test = shift_res_action_in_states(states, path=("rnn", "res", "action"))[1]
    
    # print(test)
    # print(test_state)
    # print(states[1])
    