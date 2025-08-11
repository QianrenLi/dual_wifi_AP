import matplotlib.pyplot as plt
import ast
from collections import defaultdict
from math import floor

# Sample data as a multi-line string (replace with file reading)
with open('test_mon_freq/mac-info_10ms.txt', 'r') as f:
    data = f.read()

# Organize data by integer second and ac_ind
grouped_data = defaultdict(lambda: defaultdict(lambda: {'time': [], 'packet_num': []}))

for line in data.strip().split('\n'):
    timestamp_str, dict_str = line.split(' - ', 1)
    timestamp = float(timestamp_str)
    integer_second = floor(timestamp)
    fractional = timestamp - integer_second
    
    # Parse the dictionary
    d = ast.literal_eval(dict_str)
    
    # Store data for each ac_ind
    for ac_ind, packet_num in d.items():
        grouped_data[integer_second][ac_ind]['time'].append(fractional)
        grouped_data[integer_second][ac_ind]['packet_num'].append(packet_num)

# Create separate plots for each second
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']  # Different markers for each ac_ind

for second, ac_data in grouped_data.items():
    plt.figure(figsize=(10, 6))
    plt.title(f"Second: {second}")
    plt.xlabel("Fractional Second")
    plt.ylabel("Packet Number")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, (ac_ind, data_dict) in enumerate(ac_data.items()):
        # Sort by time to ensure proper line connections
        sorted_data = sorted(zip(data_dict['time'], data_dict['packet_num']))
        times, packets = zip(*sorted_data)
        
        # Plot with unique marker and line style
        plt.plot(
            times,
            packets,
            label=f'ac_ind {ac_ind}',
            marker=markers[i % len(markers)],
            markersize=8,
            linestyle='-',
            linewidth=2
        )
    
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"test_mon_freq/10ms/plot_second_{second}.png")