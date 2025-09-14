import time
import os
from datetime import datetime

from util.exp_setup import create_transmission_config
from util.flows import flow_to_rtt_log
from tap import Connector

conn = Connector()
duration = 5
exp_name = "local_exp"

tx_srcs, flows = create_transmission_config(exp_name, conn, is_update=True)

start_time = time.time()

## Rx
for port, flow in flows.items():
    if 'file' in flow.npy_file:
        conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port})
    else:
        conn.batch(flow.dst_sta, "receive_file", {"duration": duration, "port": port, 'hyper_parameters': f'--calc-rtt --src-ipaddrs {flow.tx_ipaddrs[0]}'})

## Tx
for tx, srcs in tx_srcs.items():
    print(f"Transmission: {tx}")
    for src in srcs:
        config_path = "/".join(src.split("/")[1:])
        conn.batch(tx, "send_file", {"duration": duration, "config": config_path})

## Get Result
conn.executor.fetch()
while True:
    try:
        res = conn.apply()
        break
    except Exception as e:
        time.sleep(1)

res = [r for r in res if r != {}]
print(res)

## Pull RTT logs
log_dir = "stream-replay/logs"

folder = f'exp_trace/{exp_name}/trial_{datetime.now().strftime("%Y%m%d-%H%M")}'

for flow in flows.values():
    client = flow.src_sta
    file_name = flow_to_rtt_log(flow)
    Connector(client).sync_file(f'{log_dir}/{file_name}')
    # rename
    os.makedirs(f'{folder}', exist_ok=True)
    os.rename(f'{log_dir}/{file_name}', f'{folder}/{file_name}')
    
for tx, srcs in tx_srcs.items():
    Connector(tx).sync_file('stream-replay/logs/recorder.txt')
    os.rename('stream-replay/logs/recorder.txt', f'{folder}/recorder_{tx}.txt')

print( "Execution time: ", time.time() - start_time)