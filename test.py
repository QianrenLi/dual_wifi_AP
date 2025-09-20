import time
import os
import argparse
from datetime import datetime
from util.exp_setup import create_transmission_config
from util.flows import flow_to_rtt_log
from tap import Connector

def apply_exist_command(conn:Connector):
    while True:
        try:
            outputs = conn.apply()
            break
        except:
            continue
    return outputs

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the transmission experiment.")
    parser.add_argument("--duration", type=int, default=5, help="Duration of the experiment.")
    parser.add_argument("--exp_name", type=str, default="local_exp", help="Experiment name.")
    parser.add_argument("--render", action='store_true', help="Flag to render the experiment.")
    args = parser.parse_args()

    # Initialize
    conn = Connector()
    duration = args.duration
    exp_name = args.exp_name

    tx_srcs, flows = create_transmission_config(exp_name, conn, is_update=True)
    exit()
    
    start_time = time.time()
    
    ## Start agent
    src_flows = {}
    for port, flow in flows.items():
        if flow.src_sta not in src_flows:
            src_flows[flow.src_sta] = []
        src_flows[flow.src_sta].append(flow.flow_name)
    
    for tx, srcs in tx_srcs.items():
        for src in srcs:
            print(src, tx)
            conn.batch(tx, "start_agent", {"duration": duration, "config_file": src}, timeout = duration + 10).wait(0.1)
    
    # # Rx
    for port, flow in flows.items():
        if 'file' in flow.npy_file:
            conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port})
        else:
            if args.render:
                ## Wait due to the late start of xterm
                conn.batch(flow.dst_sta, "receive_file_gui", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]} --rx-mode'}, timeout = duration + 5).wait(1)
            else:
                conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]}'}, timeout = duration + 5).wait(0.1)

    # # Tx
    for tx, srcs in tx_srcs.items():
        print(f"Transmission: {tx}")
        for src in srcs:
            config_path = "/".join(src.split("/")[1:])
            conn.batch(tx, "send_file", {"duration": duration, "config": config_path}, timeout = duration + 5)

    # Get Result
    conn.executor.fetch()
    while True:
        try:
            res = conn.apply()
            break
        except Exception as e:
            time.sleep(1)

    res = [r for r in res if r != {}]
    print(res)

    # Pull RTT logs
    log_dir = "stream-replay/logs"

    folder = f'exp_trace/{exp_name}/trial_{datetime.now().strftime("%Y%m%d-%H%M")}'
    for flow in flows.values():
        client = flow.src_sta
        file_name = flow_to_rtt_log(flow)
        Connector(client).sync_file(f'{log_dir}/{file_name}')
        os.makedirs(f'{folder}', exist_ok=True)
        os.rename(f'{log_dir}/{file_name}', f'{folder}/{file_name}')
        
    for tx, srcs in tx_srcs.items():
        # Connector(tx).sync_file('logs/agent/rollout.jsonl')
        os.rename('logs/agent/rollout.jsonl', f'{folder}/rollout.jsonl')

    print("Execution time:", time.time() - start_time)

if __name__ == "__main__":
    main()
