from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Link:
    src_ip: str
    dst_ip: str

@dataclass
class Flow:
    src_sta: str
    dst_sta: str
    window_size: int
    tx_ipaddrs: List[str]
    type: str
    npy_file: str
    tos: int
    port: int
    throttle: int
    start_offset: int
    priority: str
    links: List[Link]
    tx_part: float
    calc_rtt: bool
    no_logging: bool
    
    @property
    def flow_name(self) -> str:
        return f"{self.port}@{self.tos}"

def reshape_to_flows_by_port(cfg: Dict[str, Dict[str, dict]]) -> Dict[int, Flow]:
    """
    Convert nested {src_sta: {dst_sta: {...}}} to {port: Flow}.
    Assumes port numbers are globally unique; raises on duplicates.
    """
    flows: Dict[int, Flow] = {}
    for src_sta, dst_map in cfg.items():
        for dst_sta, params in dst_map.items():
            window_size = int(params["window_size"])
            tx_ipaddrs = list(params["tx_ipaddrs"])
            for s in params["streams"]:
                port = int(s["port"])
                if port in flows:
                    raise ValueError(
                        f"Duplicate port {port} (existing {flows[port].src_sta}->{flows[port].dst_sta}, "
                        f"new {src_sta}->{dst_sta}). Make ports unique."
                    )
                flow = Flow(
                    src_sta=src_sta,
                    dst_sta=dst_sta,
                    window_size=window_size,
                    tx_ipaddrs=tx_ipaddrs,
                    type=s["type"],
                    npy_file=s["npy_file"],
                    tos=int(s["tos"]),
                    port=port,
                    throttle=int(s.get("throttle", 0)),
                    start_offset=int(s.get("start_offset", 0)),
                    priority=str(s.get("priority", "")),
                    links=[Link(a, b) for (a, b) in s["links"]],
                    tx_part=float(s.get("tx_part", 1.0)),
                    calc_rtt=bool(s.get("calc_rtt", False)),
                    no_logging=bool(s.get("no_logging", False)),
                )
                flows[port] = flow
    return flows

def flow_to_rtt_log(flow: Flow) -> str:
    """
    Generate a log filename for RTT based on the flow's properties.
    """
    return f"rtt-{flow.port}@{flow.tos}.txt"