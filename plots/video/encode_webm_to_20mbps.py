#!/usr/bin/env python3
"""
Encode webm videos to 20 Mbps bitrate
"""
from encoder import encode_multiversion_slots
from pathlib import Path
import json
import argparse

def encode_video(input_path, output_dir, fps=None, num_slots=None, t_min=None, t_max=None):
    """Encode a webm video to 20 Mbps with 1-second slots, optionally with time restrictions

    Args:
        input_path: Path to input webm video file
        output_dir: Output directory for encoded video
        fps: Target frames per second (if None, use source fps)
        num_slots: Number of 1-second slots to encode (optional)
        t_min: Minimum time (seconds) to start encoding from (optional)
        t_max: Maximum time (seconds) to encode up to (optional)
    """
    import cv2

    # Get video properties first
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {input_path}")
        return None

    fps_source = cap.get(cv2.CAP_PROP_FPS)
    width_source = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_source = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps_source if fps_source > 0 else 0
    cap.release()

    print(f"Video info: {width_source}x{height_source}, {fps_source:.2f} fps, {duration:.2f}s duration")

    # Use source fps if not specified
    if fps is None:
        fps = int(round(fps_source))
        print(f"Using source fps: {fps}")

    # Calculate number of slots based on time restrictions
    if t_min is not None or t_max is not None:
        # Calculate time range
        start_time = t_min if t_min is not None else 0
        end_time = t_max if t_max is not None else duration

        # Ensure valid range
        if start_time >= duration:
            print(f"Warning: t_min ({start_time}s) >= video duration ({duration:.2f}s)")
            return None

        if end_time > duration:
            print(f"Warning: t_max ({end_time}s) > video duration ({duration:.2f}s), adjusting to {duration:.2f}s")
            end_time = duration

        # Calculate number of slots to encode
        duration_to_encode = end_time - start_time
        num_slots_calculated = int(duration_to_encode)

        print(f"Encoding from {start_time}s to {end_time}s ({duration_to_encode:.2f}s total)")
        print(f"Number of slots to encode: {num_slots_calculated}")

        num_slots = num_slots_calculated

        # Note: The encoder.py API doesn't directly support start time offset
        # We would need to modify it to skip frames initially, but for now
        # we'll encode from the beginning and limit the number of slots

    return encode_multiversion_slots(
        input_file=input_path,
        out_dir=output_dir,
        bitrates_bps=[20_000_000],  # 20 Mbps
        slot_seconds=1,
        target_fps=fps,
        width=width_source,  # Use source width
        height=height_source,  # Use source height
        num_slots=num_slots
    )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Encode webm videos to 20 Mbps bitrate with optional time range')
    parser.add_argument('input_files', nargs='+',
                        help='Input webm video files to encode')
    parser.add_argument('--t_min', type=float, default=None,
                        help='Minimum time to start encoding from (seconds)')
    parser.add_argument('--t_max', type=float, default=None,
                        help='Maximum time to encode up to (seconds)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to add to output directory names')

    args = parser.parse_args()

    # Process each input file
    for input_file in args.input_files:
        input_path = Path(input_file)

        # Check if file exists
        if not input_path.exists():
            print(f"Error: Video file not found: {input_path}")
            continue

        # Check if it's a webm file
        if input_path.suffix.lower() != '.webm':
            print(f"Warning: {input_path} is not a webm file. Skipping.")
            continue

        # Create output directory name based on input file
        base_name = input_path.stem
        suffix_part = f"_{args.suffix}" if args.suffix else ""
        output_dir = Path(f"encoded_videos/{base_name}_20mbps{suffix_part}")

        print(f"\nEncoding {input_path}...")
        manifest = encode_video(
            input_path=str(input_path),
            output_dir=str(output_dir),
            fps = 60,
            t_min=args.t_min,
            t_max=args.t_max
        )

        if manifest is None:
            print(f"Failed to encode {input_path}")
            continue

        # Save manifest for reference
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Print time range information
        time_info = ""
        if args.t_min is not None or args.t_max is not None:
            time_range = []
            if args.t_min is not None:
                time_range.append(f"from {args.t_min}s")
            else:
                time_range.append("from start")

            if args.t_max is not None:
                time_range.append(f"to {args.t_max}s")
            else:
                time_range.append("to end")

            time_info = f" ({' '.join(time_range)})"

        print(f"Encoded to: {output_dir}{time_info}")
        print(f"  - Slots: {manifest['slots']}")
        print(f"  - Duration: {manifest['slots']} seconds")
        print(f"  - Resolution: {manifest['width']}x{manifest['height']}")
        print(f"  - FPS: {manifest['fps']}")
        print(f"  - Bitrate: {manifest['bitrates_bps'][0] / 1e6:.1f} Mbps")

if __name__ == "__main__":
    main()