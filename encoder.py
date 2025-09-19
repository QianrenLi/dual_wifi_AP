import json
import math
import struct
from pathlib import Path
import av.packet
import cv2
import av

def _make_encoder(width, height, fps, bitrate_bps, gop):
    codec = av.CodecContext.create('h264', 'w')
    codec.width = width
    codec.height = height
    codec.pix_fmt = 'yuv420p'
    codec.framerate = fps
    codec.bit_rate = int(bitrate_bps)
    x264_params = (
        f"keyint={gop}:min-keyint={gop}:scenecut=0:"
        f"bframes=0:repeat-headers=1:aud=1:rc-lookahead=0"
    )
    
    codec.options = {
        'preset': 'fast',
        'tune': 'zerolatency',
        'g': str(gop),
        'keyint_min': str(gop),
        'scenecut': '0',
        'bf': '0',
        'b_strategy': '0',
        'annexb': '1',  
        'x264-params': x264_params,
    }
    
    codec.options['force_key_frames'] = '0'
    return codec

def _write_slot_file(path, interval_ns, frame_packets):
    """Write one slot to a .bin file: header + [len|data] per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        # Write header: [interval_ns, len(data)]
        for pkt in frame_packets:
            data = bytes(pkt)
            f.write(struct.pack('>QQ', interval_ns, len(data))) 
            f.write(data)

def encode_multiversion_slots(
    input_file: str,
    out_dir: str,
    bitrates_bps,           # e.g., [5_000_000, 2_500_000, 1_000_000]
    slot_seconds: int,      # e.g., 1
    target_fps: int = None,  # if None, use source fps
    width: int = None,
    height: int = None,
    check_decode: bool = False,
    num_slots = None,
):
    """
    Produce per-bitrate folders and per-slot .bin files.
    Now uses a single encoder per version (continuous GOP/RD across slots).
    """
    out_dir = Path(out_dir)
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {input_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = target_fps if target_fps else int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = width if width else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = height if height else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_slot = fps * int(slot_seconds)
    assert frames_per_slot > 0, "slot_seconds must be >= 1 (and fps > 0)."
    num_slots = num_slots if num_slots else math.ceil(total_frames / frames_per_slot) if total_frames > 0 else 0

    manifest = {
        "input": str(input_file),
        "width": width,
        "height": height,
        "fps": fps,
        "slot_seconds": slot_seconds,
        "frames_per_slot": frames_per_slot,
        "bitrates_bps": bitrates_bps,
        "slots": num_slots,
        "versions": []
    }

    interval_ns = int(1e9 / fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    decode_packets = []
    
    for v_idx, br in enumerate(bitrates_bps):
        version_label = f"v{v_idx}_{int(round(br/1000))}kbps"
        version_dir = out_dir / version_label
        version_info = {
            "label": version_label,
            "bitrate_bps": int(br),
            "dir": str(version_dir),
            "files": []
        }

        enc = _make_encoder(width, height, fps, br, gop=frames_per_slot)

        current_slot = 0
        current_slot_packets = []
        frames_seen = 0
        wrote_any = False

        def write_slot_and_reset(slot_index, packets):
            nonlocal wrote_any
            if not packets:
                return
            slot_path = version_dir / f"seg_{slot_index:06d}.bin"

            print(f"[INFO] Writing to {slot_path} ({len(packets)} packets)")
            _write_slot_file(slot_path, interval_ns, packets)

            start_frame = slot_index * frames_per_slot
            end_frame = start_frame + frames_per_slot if total_frames == 0 else min(start_frame + frames_per_slot, total_frames)
            version_info["files"].append({
                "slot_index": slot_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "path": str(slot_path)
            })
            wrote_any = True

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            
            pkts = enc.encode(av_frame)
            decode_packets.append(b''.join(bytes(pkt) for pkt in pkts))

            slot_idx_for_frame = frames_seen // frames_per_slot
            if slot_idx_for_frame != current_slot:
                for p in enc.encode(None):
                    current_slot_packets.append(p)
                write_slot_and_reset(current_slot, current_slot_packets)
                enc = _make_encoder(width, height, fps, br, gop=frames_per_slot)
                current_slot_packets = []
                current_slot = slot_idx_for_frame

            for p in pkts:
                current_slot_packets.append(p)

            frames_seen += 1
            if num_slots and current_slot >= num_slots:
                break

        for p in enc.encode(None):
            current_slot_packets.append(p)

        if current_slot < num_slots:
            write_slot_and_reset(current_slot, current_slot_packets)

        if check_decode:
            codec = av.CodecContext.create('h264', 'r')
            win = "Decoded Frame"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

            try:
                for data in decode_packets:
                    if not data:
                        continue
                    packet = av.packet.Packet(data)
                    frames = codec.decode(packet)
                    for frame in frames:
                        img = frame.to_ndarray(format='bgr24')
                        img = cv2.resize(img, (1280, 720))
                        cv2.imshow(win, img)
                        if cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                frames = codec.decode(None)
                for frame in frames:
                    img = frame.to_ndarray(format='bgr24')
                    img = cv2.resize(img, (1280, 720))
                    cv2.imshow(win, img)
                    if cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF == ord('q'):
                        break

            except KeyboardInterrupt:
                pass
            except Exception as e:
                print(f"[ERROR] Decoding check failed for version {version_label}: {e}")
            finally:
                cv2.destroyWindow(win)

        manifest["versions"].append(version_info)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cap.release()
    return manifest

if __name__ == '__main__':
    # Example:
    manifest = encode_multiversion_slots(
        input_file="video/4096_2160_60fps.mp4",
        out_dir="stream-replay/data/video",
        bitrates_bps=list(range(2_000_000, 32_000_000, 2_000_000)),
        # bitrates_bps = [10_000_000],
        slot_seconds = 1,
        target_fps = 60,
        width=4096,
        height=2160,
        num_slots = 18,
    )
    print("Wrote manifest to:", Path("stream-replay/data/video/manifest.json").resolve())