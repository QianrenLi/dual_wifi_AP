import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
queue_data_path = '4K_5G/mac_queue.csv'
rtt_data_path = '4K_5G/rtt.txt'  # Assuming the RTT file is in the same directory
data_folder = os.path.dirname(queue_data_path)

# Read and process queue data
df_queue = pd.read_csv(queue_data_path)
df_queue['timestamp'] = pd.to_datetime(df_queue['timestamp'], unit='ns')

# Create figure with two subplots
plt.figure(figsize=(12, 12))

# First subplot - Q0 packet count
plt.subplot(2, 1, 1)
df_queue.plot(x='timestamp', y='Q0_pkt_num', ax=plt.gca())
plt.title('Q0 Packet Count Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Packet Count')

# Read and process RTT data
df_rtt = pd.read_csv(rtt_data_path, sep=' ', header=None, names=['frame_id', 'rtt_value', 'queue'], index_col=False)
df_rtt['frame_id'] = pd.to_numeric(df_rtt['frame_id'])
df_rtt['rtt_value'] = pd.to_numeric(df_rtt['rtt_value']) * 1000

print(df_rtt)

# Second subplot - RTT values
plt.subplot(2, 1, 2)
df_rtt.plot(x='frame_id', y='rtt_value', ax=plt.gca(), style='.-')
plt.title('RTT Values per Frame')
plt.xlabel('Frame ID')
plt.ylabel('RTT (seconds)')

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'{data_folder}/combined_metrics.png')

# Show separate figures as well (optional)
# Figure 1: Queue data
plt.figure(figsize=(12, 6))
df_queue.plot(x='timestamp', y='Q0_pkt_num')
plt.title('Q0 Packet Count Over Time')
plt.savefig(f'{data_folder}/q0_packet_count_over_time.png')

# Figure 2: RTT data
plt.figure(figsize=(12, 6))
df_rtt.plot(x='frame_id', y='rtt_value', style='.-')
plt.title('RTT Values per Frame')
plt.savefig(f'{data_folder}/rtt_values.png')

plt.close('all')  # Close all figures