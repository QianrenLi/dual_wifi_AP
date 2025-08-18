import time
import os

from util.exp_setup import create_transmission_config
from util.flows import flow_to_rtt_log
from tap import Connector

conn = Connector()
duration = 5
exp_name = "local_exp"

tx_srcs, flows = create_transmission_config("local_exp", conn, is_update=True)

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
for client in conn.list_all():
    file_name = flow_to_rtt_log(flow)
    Connector(client).sync_file(f'{log_dir}/{file_name}')
    # rename
    os.makedirs(f'exp_trace/{exp_name}', exist_ok=True)
    os.rename(f'{log_dir}/{file_name}', f'exp_trace/{exp_name}/{file_name}')

print( "Execution time: ", time.time() - start_time)