import importlib.util
import sys
import os
import json
import time

from pathlib import Path
from typing import Dict, Tuple

from util.flows import reshape_to_flows_by_port, Flow

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tap import Connector

TxSrcs = Dict[str, Dict[str, str]]        # tx -> {"control_config": "...", "transmission_config": "..."}
Flows = Dict[int, Flow]                   # port -> Flow

def load_config_file(config_name):
    config_path = Path(config_name)
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg

def create_transmission_config(config_name, exp_name, conn: Connector, is_update=False, duration = None) -> Tuple[TxSrcs, Flows, TxSrcs, Flows]:
    cfg = load_config_file(config_name)
    macs_need_separate = cfg.macs_need_separate
    clients = Connector().list_all()
    while True:
        try:
            for client in clients:
                conn.batch(client, 'read_ip_addr')
            outputs = conn.executor.wait(1).fetch().apply()
            break
        except Exception as e:
            print("Script Assign Fail", e)
            time.sleep(1)
        
    results = [o["ip_addr"] for o in outputs]
    ip_table = {}
    for name, result in zip(clients, results):
        ip_addr = eval(result)
        ip_table[name] = ip_addr
    
    configs = cfg.exp_streams(ip_table); policy_configs =cfg.policy_config()
    flows = reshape_to_flows_by_port(configs)
    
    tx_srcs = {}
    
    tx_folder = f"stream-replay/data/configs/{exp_name}"
    os.makedirs(tx_folder, exist_ok=True)
    network_folder = f"net_util/net_config/{exp_name}"
    os.makedirs(network_folder, exist_ok=True)
    
    for src, config in configs.items():
        tx_srcs[src] = {}
        ## assert only one dest in config
        assert len(config) == 1, "Only support one dest in config"
        for dest, stream_config in config.items():
            data_path = f"{tx_folder}/{src}_{dest}.json"
            with open(data_path, "w") as f:
                f.write(json.dumps(stream_config, indent=4))
            tx_srcs[src]['transmission_config'] = data_path
            
            data_path = f"{network_folder}/{src}_{dest}.json"
            with open(data_path, "w") as f:
                if duration:
                    policy_configs[src][dest]['agent_cfg']['duration'] = duration
                f.write(json.dumps(policy_configs[src][dest], indent=4))
            tx_srcs[src]['control_config'] = data_path
    
    ## Create interference source
    inter_src = {}
    inter_flow = {}
    if hasattr(cfg, "interference_streams") and callable(getattr(cfg, "interference_streams")):
        inter_configs = cfg.interference_streams(ip_table)
        inter_flow = reshape_to_flows_by_port(inter_configs)
        for src, config in inter_configs.items():
            inter_src[src] = {}
            ## assert only one dest in config
            assert len(config) == 1, "Only support one dest in config"
            for dest, stream_config in config.items():
                data_path = f"{tx_folder}/{src}_{dest}.json"
                with open(data_path, "w") as f:
                    f.write(json.dumps(stream_config, indent=4))
                inter_src[src]['transmission_config'] = data_path
            
    if is_update:         
        ## Sync the config_name folder to all clients
        for client in clients:
            Connector(client).sync_file(tx_folder, is_pull=False)
            Connector(client).sync_file(network_folder, is_pull=False)
            
    return tx_srcs, flows, inter_src, inter_flow, macs_need_separate

if __name__ == "__main__":
    # conn = Connector()
    # for client in conn.list_all():
    #     Connector(client).sync_file("stream-replay/logs/test.txt", is_pull=False)
        
    # create_transmission_config("system_verify", conn)
    cfg = load_config_file("system_verify")
    policy_cfg = cfg.policy_config()
    
    for src, config in policy_cfg.items():
        for dest, per_policy_config in config.items():
            data_path = f"net_util/net_config/{src}_{dest}.json"
            with open(data_path, "w") as f:
                f.write(json.dumps(per_policy_config, indent=4))
    
    