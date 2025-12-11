import importlib
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
    """
    Load a configuration file that may use relative imports.

    This function handles both standalone config files and files that use
    relative imports within a package structure.
    """
    config_path = Path(config_name).resolve()

    # If the file is in a package structure (has __init__.py), use import_module
    config_dir = config_path.parent

    # Check if this is part of a package
    init_file = config_dir / "__init__.py"
    if init_file.exists():
        # It's a package - we need to import it properly
        # Build the full module path from the root
        relative_path = config_path.relative_to(Path.cwd())
        module_parts = list(relative_path.parts)

        # Remove the .py extension from the last part
        module_parts[-1] = module_parts[-1].replace('.py', '')

        # Build module path
        module_path = '.'.join(module_parts)

        try:
            # Use import_module to properly handle relative imports
            cfg = importlib.import_module(module_path)
            return cfg
        except ImportError as e:
            print(f"Warning: Failed to import {module_path}: {e}")
            print("Falling back to direct file loading...")

    # Fallback to direct loading for simple modules or when import fails
    # Save original sys.path
    original_path = sys.path[:]

    try:
        # Insert the config directory and its parent at the beginning of sys.path
        # This allows Python to resolve relative imports properly
        config_dir_str = str(config_dir)
        parent_config_dir = str(config_dir.parent)

        if config_dir_str not in sys.path:
            sys.path.insert(0, config_dir_str)
        if parent_config_dir not in sys.path:
            sys.path.insert(0, parent_config_dir)

        # Determine module name from the file path
        # Remove the .py extension and use the filename as module name
        module_name = config_path.stem

        # Create spec with the proper module name that includes package context
        # This allows relative imports to work
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        cfg = importlib.util.module_from_spec(spec)

        # Set the package attribute to enable relative imports
        # This is the key fix - Python needs to know what package this module belongs to
        cfg.__package__ = config_dir.name

        # Execute the module
        spec.loader.exec_module(cfg)
        return cfg
    finally:
        # Restore original sys.path
        sys.path = original_path

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
    
    