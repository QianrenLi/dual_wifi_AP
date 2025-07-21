import cv2
import av
import struct

def encode_and_store_video(input_file, output_file, width=1280, height=720):
    # Video Capture
    cap = cv2.VideoCapture(input_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # H.264 Encoder Configuration
    codec = av.CodecContext.create('h264', 'w')
    codec.width = width
    codec.height = height
    codec.pix_fmt = 'yuv420p'
    codec.framerate = fps
    codec.options = {
        'preset': 'fast',
        'tune': 'zerolatency',
        'crf': '23',
        'g': str(int(fps)),
    }
    
    stored_packets = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        packets = codec.encode(av_frame)
        
        datas = []
        for packet in packets:
            data = bytes(packet)
            datas.append(data)
        
        data = b''.join(datas)
        stored_packets.append(data)
        frame_count += 1
    
    # Flush encoder
    packets = codec.encode(None)
    for packet in packets:
        try:
            data = bytes(packet)
        except TypeError:
            data = packet.to_bytes()
        stored_packets.append(data)
    
    cap.release()
    
    # Save stored packets to file
    with open(output_file, 'wb') as f:
        for frame_count, data in enumerate(stored_packets):
            # Write frame count and data length as metadata
            f.write(struct.pack('>Q', len(data)))
            f.write(data)
    
    return stored_packets

# encode_and_store_video('video/gn.mp4', 'video/gn.bin')
encode_and_store_video('video/4096_2160_25fps.mp4', 'video/4096_2160_25fps.bin', width=4096, height=2160)