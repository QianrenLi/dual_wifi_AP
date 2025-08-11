#!/usr/bin/env python3
import time
import os
import csv
import re
from collections import OrderedDict
import subprocess as sp

# Configuration
PROC_FILE = "/proc/net/rtl88XXau/wlx081f7165e561/mac_qinfo"
OUTPUT_FILE = f"mac_qinfo_{time.strftime('%Y%m%d_%H%M%S')}.csv"
SAMPLE_INTERVAL = 0.005  # 5ms
QUEUE_HEADERS = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'MG', 'HI', 'BCN']

SHELL_POPEN = lambda x: sp.Popen(x, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
SHELL_RUN = lambda x: sp.run(x, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)

def clean_value(val):
    """Remove commas and trailing punctuation from values"""
    return val.rstrip(', ')

def parse_queue_line(line):
    """Parse a single queue line into a dictionary of key-value pairs"""
    result = {}
    # Extract key-value pairs using regex
    matches = re.findall(r'(\w+):([^\s,]+)', line)
    for key, val in matches:
        result[key] = clean_value(val)
    return result

def parse_queue_data(data):
    """Parse all queue data into a structured dictionary"""
    queues = OrderedDict()
    
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Identify queue type
        for qid in QUEUE_HEADERS:
            if line.startswith(qid):
                queues[qid] = parse_queue_line(line)
                break
                
    return queues

def main():
    # Verify proc file exists
    if not os.path.exists(PROC_FILE):
        print(f"Error: {PROC_FILE} not found!")
        exit(1)
    
    # Prepare CSV output
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        # Build CSV header
        fieldnames = ['timestamp']
        for qid in QUEUE_HEADERS:
            if qid == 'BCN':
                fieldnames.extend([f'{qid}_head', f'{qid}_pkt_num'])
            else:
                fieldnames.extend([
                    f'{qid}_head', 
                    f'{qid}_tail', 
                    f'{qid}_pkt_num',
                    f'{qid}_macid',
                    f'{qid}_ac'
                ])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"Monitoring MAC queues at {1/SAMPLE_INTERVAL:.0f}Hz. Press Ctrl-C to stop...")
        
        try:
            while True:
                start_time = time.perf_counter()
                timestamp = time.time_ns()
                
                # Read queue information
                with open(PROC_FILE, 'r') as f:
                    data = f.read().strip()
                
                # Parse queue data
                queues = parse_queue_data(data)
                
                # Build CSV row
                row = {'timestamp': timestamp}
                for qid, qdata in queues.items():
                    if qid == 'BCN':
                        row[f'{qid}_head'] = qdata.get('head', 'N/A')
                        row[f'{qid}_pkt_num'] = qdata.get('pkt_num', 'N/A')
                    else:
                        row[f'{qid}_head'] = qdata.get('head', 'N/A')
                        row[f'{qid}_tail'] = qdata.get('tail', 'N/A')
                        row[f'{qid}_pkt_num'] = qdata.get('pkt_num', 'N/A')
                        row[f'{qid}_macid'] = qdata.get('macid', 'N/A')
                        row[f'{qid}_ac'] = qdata.get('ac', 'N/A')
                
                # Write to CSV
                writer.writerow(row)
                csvfile.flush()  # Ensure immediate write
                
                # Precise sleep timing
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, SAMPLE_INTERVAL - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    SHELL_POPEN("cd stream-replay; cargo run --bin stream-replay-tx data/single_ap_test.json 30")
    main()