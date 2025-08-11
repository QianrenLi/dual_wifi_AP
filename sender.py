import socket
import struct
import time
import math


def send_packets(packets_file, udp_ip, udp_port, target_fps=30):
    MAX_UDP_SIZE = 1472  # Safe payload size for UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Read stored packets from file
    stored_packets = []
    with open(packets_file, 'rb') as f:
        while True:
            # Read metadata (frame_count and data length)
            metadata = f.read(8)
            if not metadata:
                break
            data_length = struct.unpack('>Q', metadata)
            # Read the actual data
            data = f.read(data_length[0])
            stored_packets.append(data)
    
    frame_interval = 1.0 / target_fps
    last_frame_time = None
    total_bytes_sent = 0
    
    start_time = time.time()
    
    for frame_count, data in enumerate(stored_packets):
        # Split large frames into chunks
        total_chunks = math.ceil(len(data) / MAX_UDP_SIZE)
        for chunk_idx in range(total_chunks):
            start = chunk_idx * MAX_UDP_SIZE
            end = start + MAX_UDP_SIZE
            chunk = data[start:end]
            
            # Header: frame_count, chunk_idx, total_chunks, chunk_size
            header = struct.pack('>IIII', frame_count, chunk_idx, total_chunks, len(chunk))
            
            sock.sendto(header + chunk, (udp_ip, udp_port))
            total_bytes_sent += len(header) + len(chunk)
        
        elapsed_time = time.time() - start_time
        mbps = (total_bytes_sent * 8) / (elapsed_time * 1_000_000)
        
        # Frame rate control
        if last_frame_time is not None:
            elapsed = time.time() - last_frame_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        last_frame_time = time.time()
        print(f"Frame {frame_count} | {mbps:.2f} Mbps | Sent at {time.strftime('%H:%M:%S')}")
    
    sock.close()

send_packets('video/1280_720_25fps.bin', "192.168.3.8", 5005, 25)