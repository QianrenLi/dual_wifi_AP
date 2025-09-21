import importlib.util
from pathlib import Path
import sys
import os
import json

from util.flows import reshape_to_flows_by_port

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tap import Connector

def load_config_file(config_name):
    config_path = Path("config") / f"{config_name}.py"
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg

def create_transmission_config(config_name, conn: Connector, is_update=False):
    cfg = load_config_file(config_name)
    clients = Connector().list_all()
    for client in clients:
        conn.batch(client, 'read_ip_addr')
    outputs = conn.executor.wait(1).fetch().apply()
    results = [o["ip_addr"] for o in outputs]
    ip_table = {}
    for name, result in zip(clients, results):
        ip_addr = eval(result)
        ip_table[name] = ip_addr
    
    configs = cfg.exp_streams(ip_table); policy_configs =cfg.policy_config()
    flows = reshape_to_flows_by_port(configs)
    
    tx_srcs = {}
    
    tx_folder = f"stream-replay/data/configs/{config_name}"
    os.makedirs(tx_folder, exist_ok=True)
    network_folder = f"net_util/net_config/{config_name}"
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
                f.write(json.dumps(policy_configs[src][dest], indent=4))
            tx_srcs[src]['control_config'] = data_path

    if is_update:         
        ## Sync the config_name folder to all clients
        for client in clients:
            Connector(client).sync_file(tx_folder, is_pull=False)
            Connector(client).sync_file(network_folder, is_pull=False)
            
    return tx_srcs, flows

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
    
    