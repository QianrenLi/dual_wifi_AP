import socket
import heapq
import av
import cv2
import struct
import time
from collections import defaultdict, deque
from threading import Thread, Lock

# UDP Configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Video Configuration
BUFFER_SIZE = 10  # Number of frames to buffer
MAX_FRAME_AGE = 0.060  # 60ms in seconds
TARGET_FPS = 25  # Target display frame rate

# H.264 Decoder Setup
codec = av.CodecContext.create('h264', 'r')

# Thread-safe frame buffer
class FrameBuffer:
    def __init__(self, fps=TARGET_FPS):
        self.heap = []  # Min-heap: (frame_count, frame)
        self.lock = Lock()
        self.last_display_time = None
        self.fps = fps
        self.to_display_frame_id = 0
        self.buffer_size = BUFFER_SIZE

    def add_frame(self, frame, frame_count):
        # Only add frames that are at least the current display ID
        if frame_count < self.to_display_frame_id:
            return
            
        with self.lock:
            heapq.heappush(self.heap, (frame_count, frame))

    def get_frame(self):
        with self.lock:                
            # Return next frame if available
            if self.heap and self.heap[0][0] == self.to_display_frame_id:
                return heapq.heappop(self.heap)[1]  # Return frame only
        return None

# Global frame buffer
frame_buffer = FrameBuffer()

# Frame reconstruction buffers
frame_chunks = defaultdict(dict)

def udp_receiver():
    while True:
        data, _ = sock.recvfrom(65536)
        
        # Parse header: frame_count, chunk_idx, total_chunks, chunk_size
        header = data[:16]
        frame_count, chunk_idx, total_chunks, chunk_size = struct.unpack('>IIII', header)
        payload = data[16:16+chunk_size]
        
        # Store chunk
        frame_chunks[frame_count][chunk_idx] = payload
        
        # Check if we have all chunks
        if len(frame_chunks[frame_count]) == total_chunks:
            # Reassemble frame
            full_frame = b''.join([frame_chunks[frame_count][i] 
                                 for i in sorted(frame_chunks[frame_count].keys())])
            
            try:
                packet = av.packet.Packet(full_frame)
                frames = codec.decode(packet)
                
                for frame in frames:
                    frame = frame.to_ndarray(format='bgr24')
                    frame_buffer.add_frame(frame, frame_count)
                
                del frame_chunks[frame_count]
                
            except (av.error.EOFError, av.error.InvalidDataError) as e:
                print(f"Decoding error: {e}")
                del frame_chunks[frame_count]


def video_display(target_width = 1280, target_height = 720): #width=1280, height=720
    first_frame_time = None
    
    while True:
        frame = frame_buffer.get_frame()
        if frame is not None:
            # Initialize timing on first valid frame
            if first_frame_time is None:
                first_frame_time = time.time()
            
            resized_frame = cv2.resize(frame, (target_width, target_height))
            cv2.imshow('Video Stream', resized_frame)
            # cv2.imshow('Video Stream', frame)

            frame_buffer.to_display_frame_id += 1
            frame_buffer.last_display_time = time.time()
            target_time = frame_buffer.last_display_time + (1 / frame_buffer.fps)
        
        # Calculate when this frame should be displayed
        
        if first_frame_time is not None:
            elapsed = target_time - time.time()
            if elapsed < - MAX_FRAME_AGE:
                frame_buffer.to_display_frame_id += 1
                # print(elapsed)
                print(frame_buffer.to_display_frame_id)
                target_time = target_time + (1 / frame_buffer.fps)

        # time.sleep(0.005)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

# Start threads
receiver_thread = Thread(target=udp_receiver, daemon=True)
display_thread = Thread(target=video_display, daemon=True)

receiver_thread.start()
display_thread.start()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    exit()