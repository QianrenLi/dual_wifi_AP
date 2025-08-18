import importlib.util
from pathlib import Path
import sys
import os

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

def create_log_file(folder, src, dest, stream_config):
    filename = f"{folder}/{src}_{dest}.json"
    with open(filename, "w") as f:
        f.write(stream_config.to_json())
    pass
    
    

def create_transmission_config(config_name, conn: Connector):
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
    
    folder = f"stream-replay/logs/{config_name}"
    os.makedirs(folder, exist_ok=True)
    
    configs = cfg.exp_streams(ip_table)
    for src, config in configs.items():
        for dest, stream_config in config.items():
            create_log_file(folder, src, dest, stream_config)
    
    
if __name__ == "__main__":
    conn = Connector()
    for client in conn.list_all():
        Connector(client).sync_file_pull("stream-replay/logs/test.txt")
        
    create_transmission_config("system_verify", conn)
    


