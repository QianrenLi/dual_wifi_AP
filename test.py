import time
import os
import argparse
import shutil
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

def train_loop(args, conn:Connector, tx_srcs, flows, duration, exp_name):
    traces = []
    maximum_trace_len = 5
    
    iteration = 0 #TODO: refresh the experiment setup
    while True:
        start_time = time.time()
        ## Start agent
        src_flows = {}
        for port, flow in flows.items():
            if flow.src_sta not in src_flows:
                src_flows[flow.src_sta] = []
            src_flows[flow.src_sta].append(flow.flow_name)
        
        for tx, srcs in tx_srcs.items():
            conn.batch(tx, "start_agent", {"control_config": srcs['control_config'], "transmission_config": srcs['transmission_config']})
        
        ## Rx
        wait_time = 0.1
        for idx, (port, flow) in enumerate(flows.items()):
            if idx == len(flows) - 1:
                wait_time = 3
            if 'file' in flow.npy_file:
                conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port}).wait(wait_time)
            else:
                if args.render:
                    ## Wait due to the late start of xterm
                    conn.batch(flow.dst_sta, "receive_file_gui", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]} --rx-mode'}).wait(wait_time)
                else:
                    conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]}'}).wait(wait_time)

        # # Tx
        for tx, srcs in tx_srcs.items():
            print(f"Transmission: {tx}")
            config_path = "/".join(srcs['transmission_config'].split("/")[1:])
            conn.batch(tx, "send_file", {"duration": duration, "config": config_path})

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
        exp_time = time.time()
        
        # Pull RTT logs
        log_dir = "stream-replay/logs"

        folder = f'exp_trace/{exp_name}/trial_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        # --- CLEAN STEP (run only on the very first iteration) ---
        model_dir = f'net_util/net_cp/{exp_name}'
        if iteration == 0:
            # remove the per-trial folder (if it oddly exists) and the model dir
            for p in (f'exp_trace/{exp_name}', model_dir):
                if os.path.exists(p):
                    print(f"[clean] removing {p}")
                    shutil.rmtree(p, ignore_errors=True)
        # ---------------------------------------------------------
        
        for flow in flows.values():
            client = flow.src_sta
            file_name = flow_to_rtt_log(flow)
            Connector(client).sync_file(f'{log_dir}/{file_name}')
            os.makedirs(f'{folder}', exist_ok=True)
            os.rename(f'{log_dir}/{file_name}', f'{folder}/{file_name}')


            
        for tx, srcs in tx_srcs.items():
            # Connector(tx).sync_file('logs/agent/rollout.jsonl')
            os.rename('logs/agent/rollout.jsonl', f'{folder}/rollout.jsonl')
        
        ## Forward the rollout.jsonl to the Train Agent
        for tx, srcs in tx_srcs.items():
            Connector('TrainAgent').sync_file(f'{folder}/rollout.jsonl', is_pull=False)
        
        exp_sync_time = time.time()
        
        ## TODO: integrated training
        traces.append(f'{folder}/rollout.jsonl')
        if len(traces) > maximum_trace_len:
            traces.pop(0)
        
        if iteration > 0:
            conn.batch('TrainAgent', "model_train", {"control_config": srcs['control_config'], "trace_path": " ".join(traces), 'load_path': f'net_util/net_cp/{exp_name}/{iteration}.pt'})
        else:
            conn.batch('TrainAgent', "model_train", {"control_config": srcs['control_config'], "trace_path": " ".join(traces)})
        
        conn.executor.fetch()
        while True:
            try:
                res = conn.apply()
                break
            except Exception as e:
                time.sleep(1)
        
        train_time = time.time()
        
        iteration += 1
        Connector('TrainAgent').sync_file(f'net_util/net_cp/{exp_name}/{iteration}.pt', is_pull=True)
        Connector('TrainAgent').sync_file('net_util/logs/train.log', is_pull=True, timeout=1)

        ## TODO: dispatch the model to tx
    
        os.rename('net_util/logs/train.log', f'{folder}/train.log')   
        
        print(f"Iteration {iteration-1}:")
        current_time = time.time()
        print(f"Exp time {exp_time - start_time}; Sync time {exp_sync_time - exp_time}; Train Time: {train_time - exp_sync_time}; Model Pull Time: {current_time - train_time}")
        print(f"Iteration {iteration-1} Execution time:", current_time - start_time)
        
        # if iteration == 1500:
        #     exit()
        
        
        
def evaluate(args, conn, tx_srcs, flows, duration, exp_name):
    ## Start agent
    src_flows = {}
    for port, flow in flows.items():
        if flow.src_sta not in src_flows:
            src_flows[flow.src_sta] = []
        src_flows[flow.src_sta].append(flow.flow_name)
    
    for tx, srcs in tx_srcs.items():
        conn.batch(tx, "start_agent_eval", {"control_config": srcs['control_config'], "transmission_config": srcs['transmission_config']})
    
    ## Rx
    wait_time = 0.1
    for idx, (port, flow) in enumerate(flows.items()):
        if idx == len(flows) - 1:
            wait_time = 3
        if 'file' in flow.npy_file:
            conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port}).wait(wait_time)
        else:
            if args.render:
                ## Wait due to the late start of xterm
                conn.batch(flow.dst_sta, "receive_file_gui", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]} --rx-mode'}).wait(wait_time)
            else:
                conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]}'}).wait(wait_time)

    # # Tx
    for tx, srcs in tx_srcs.items():
        print(f"Transmission: {tx}")
        config_path = "/".join(srcs['transmission_config'].split("/")[1:])
        conn.batch(tx, "send_file", {"duration": duration, "config": config_path})

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
    exp_time = time.time()
    
    # Pull RTT logs
    log_dir = "stream-replay/logs"

    folder = f'exp_trace/{exp_name}/trial_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    for flow in flows.values():
        client = flow.src_sta
        file_name = flow_to_rtt_log(flow)
        Connector(client).sync_file(f'{log_dir}/{file_name}')
        os.makedirs(f'{folder}', exist_ok=True)
        os.rename(f'{log_dir}/{file_name}', f'{folder}/{file_name}')
        
    for tx, srcs in tx_srcs.items():
        # Connector(tx).sync_file('logs/agent/rollout.jsonl')
        os.rename('logs/agent/rollout.jsonl', f'{folder}/rollout.jsonl')
        

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the transmission experiment.")
    parser.add_argument("--duration", type=int, default=5, help="Duration of the experiment.")
    parser.add_argument("--exp_name", type=str, default="local_exp", help="Experiment name.")
    parser.add_argument("--render", action='store_true', help="Flag to render the experiment.")
    parser.add_argument("--evaluate", action='store_true')
    args = parser.parse_args()

    # Initialize
    conn = Connector()
    duration = args.duration
    exp_name = args.exp_name

        
    tx_srcs, flows = create_transmission_config(exp_name, conn, is_update=True)
    
    if args.evaluate:
        evaluate(args, conn, tx_srcs, flows, duration, exp_name)
    else:
        train_loop(args, conn, tx_srcs, flows, duration, exp_name)

        
        
if __name__ == "__main__":
    main()
