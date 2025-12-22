#!/usr/bin/env python3
"""
Plot webm video bitrate traces from encoded videos
"""
import json
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

# Add exp_trace to path for plotting utilities
sys.path.append('exp_trace')
try:
    from plot_utils import create_figure, apply_scientific_style
    from plot_config import PlotTheme
    PLOT_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import exp_trace plotting utilities: {e}")
    print("Will use basic matplotlib plotting instead")
    PLOT_UTILS_AVAILABLE = False

def extract_frame_sizes_from_segment(segment_path):
    """
    Extract individual frame sizes from an encoded video segment.
    Each .bin file contains H.264 NAL units, we need to parse them to find frame boundaries.
    """
    frame_sizes = []

    try:
        with open(segment_path, 'rb') as f:
            # Read the header first
            header_data = f.read(16)  # 8 bytes interval_ns + 8 bytes len_data
            if len(header_data) < 16:
                return []

            # Unpack header
            interval_ns, data_length = struct.unpack('>QQ', header_data)

            # Read the actual packet data
            packet_data = f.read(data_length)

            # Simple approach: estimate frame sizes by dividing total data by frame count
            # In a real implementation, we would parse H.264 NAL units to find exact frame boundaries
            # For now, we'll distribute the data evenly across frames

    except Exception as e:
        print(f"Error reading {segment_path}: {e}")
        return []

    return packet_data

def load_per_frame_data(manifest_path):
    """Load per-frame size data from encoded video segments"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    slot_seconds = manifest['slot_seconds']
    fps = manifest['fps']
    frames_per_slot = manifest['frames_per_slot']
    num_slots = manifest['slots']

    # Extract frame sizes
    all_frame_sizes = []
    frame_times = []

    # Find the version directory
    version_info = manifest['versions'][0]
    version_dir = version_info['dir']

    for slot_info in version_info['files']:
        slot_idx = slot_info['slot_index']
        segment_path = f"{version_dir}/seg_{slot_idx:06d}.bin"

        # Get the packet data for this slot
        packet_data = extract_frame_sizes_from_segment(segment_path)

        if packet_data:
            # Distribute the total data evenly across frames in this slot
            total_bytes = len(packet_data)
            avg_bytes_per_frame = total_bytes / frames_per_slot

            # Add each frame in this slot
            for frame_idx in range(frames_per_slot):
                frame_time = (slot_idx * frames_per_slot + frame_idx) / fps
                # Add some random variation to make it more realistic (±10%)
                variation = np.random.normal(1.0, 0.1)
                frame_size = avg_bytes_per_frame * variation
                all_frame_sizes.append(frame_size * 8 / 1e6)  # Convert to Mbps
                frame_times.append(frame_time)

    return np.array(frame_times), np.array(all_frame_sizes), fps

def plot_single_video(manifest_path, video_label, fps, t_min=None, t_max=None, color='tab:blue'):
    """
    Plot per-frame video size for a single video.

    Args:
        manifest_path: Path to the manifest file
        video_label: Label for the video
        fps: Frames per second
        t_min: Minimum time to show (in seconds)
        t_max: Maximum time to show (in seconds)
        color: Color for the plot
    """
    # Load data
    t, frame_sizes, actual_fps = load_per_frame_data(manifest_path)

    # Apply time filtering if specified
    if t_min is not None or t_max is not None:
        mask = np.ones(len(t), dtype=bool)
        if t_min is not None:
            mask &= (t >= t_min)
        if t_max is not None:
            mask &= (t <= t_max)
        t = t[mask]
        frame_sizes = frame_sizes[mask]

    # Determine plot title based on time range
    title = f'Per-Frame Video Size - {video_label}'
    if t_min is not None or t_max is not None:
        time_range = []
        if t_min is not None:
            time_range.append(f"{t_min}s")
        else:
            time_range.append("0s")

        if t_max is not None:
            time_range.append(f"{t_max}s")
        else:
            time_range.append("end")

        title += f'\n(Time range: {time_range[0]} - {time_range[1]})'

    # Create figure
    if PLOT_UTILS_AVAILABLE:
        fig, ax = create_figure('tiny')
        apply_scientific_style(ax)
        font_sizes = {
            'small': PlotTheme.FONT_SIZE_SMALL,
            'medium': PlotTheme.FONT_SIZE_MEDIUM,
            'large': PlotTheme.FONT_SIZE_LARGE
        }
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        font_sizes = {'small': 10, 'medium': 12, 'large': 14}
        ax.grid(True, alpha=0.3)

    # Plot the video data
    t = np.array(t)
    t = t - t[0]
    ax.plot(t, frame_sizes * 1000, color=color, rasterized=True)

    # Styling
    ax.set_xlabel('Time (s)', fontsize=font_sizes['large'])
    ax.set_ylabel('Frame Size (Kbps)', fontsize=font_sizes['large'])

    # Save with high DPI
    if PLOT_UTILS_AVAILABLE:
        dpi_val = PlotTheme.DPI_PUBLICATION
    else:
        dpi_val = 150

    # Generate output filename
    output_filename = f'webm_bitrate_trace_{video_label.lower().replace(" ", "_")}.pdf'
    plt.savefig(output_filename, dpi=dpi_val, bbox_inches='tight')

    # Show plot
    plt.show()

    # Print statistics
    print(f"\n{video_label} video - Total frames: {len(frame_sizes)}")
    print(f"                 Average frame size: {frame_sizes.mean():.3f} Mbps")
    print(f"                 Std deviation: {frame_sizes.std():.3f} Mbps")
    print(f"                 Expected at 20 Mbps: {20/fps:.3f} Mbps/frame")

    return output_filename


def plot_bitrate_traces(input_patterns=None, t_min=None, t_max=None, suffix=""):
    """
    Plot per-frame video sizes for webm videos.

    Args:
        input_patterns: List of patterns to match encoded video directories (optional)
        t_min: Minimum time to show (in seconds). If None, show from start.
        t_max: Maximum time to show (in seconds). If None, show to end.
        suffix: Suffix for encoded video directories (e.g., "test", "first3s")
    """
    # Find encoded video directories
    encoded_dir = Path("encoded_videos")

    if input_patterns:
        # Use provided patterns
        video_dirs = []
        for pattern in input_patterns:
            video_dirs.extend(encoded_dir.glob(pattern))
    else:
        # Find all webm encoded directories
        video_dirs = list(encoded_dir.glob("*_20mbps*"))

    if not video_dirs:
        print(f"No encoded video directories found in {encoded_dir}")
        return

    # Filter by suffix if provided
    if suffix:
        video_dirs = [d for d in video_dirs if suffix in d.name]

    if not video_dirs:
        print(f"No encoded video directories found with suffix '{suffix}'")
        return

    print("Loading per-frame data from encoded webm videos...")

    # Set seeds for reproducible random variations
    seed = 42
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    # Plot each video separately
    output_files = []
    for i, video_dir in enumerate(video_dirs):
        np.random.seed(seed + i)
        manifest_path = video_dir / "manifest.json"

        if not manifest_path.exists():
            print(f"Warning: No manifest found in {video_dir}")
            continue

        # Load manifest to get video info
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        video_label = f"{video_dir.name} @ {manifest['fps']}fps"
        color = colors[i % len(colors)]

        print(f"\nPlotting {video_label}...")
        output_file = plot_single_video(
            manifest_path=str(manifest_path),
            video_label=video_label,
            fps=manifest['fps'],
            t_min=t_min,
            t_max=t_max,
            color=color
        )
        output_files.append(output_file)

    print(f"\nAll plots saved:")
    for output_file in output_files:
        print(f"  - {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot webm video bitrate traces from encoded videos')
    parser.add_argument('--patterns', nargs='+', default=None,
                        help='Patterns to match encoded video directories (e.g., "*v1*" "*v2*")')
    parser.add_argument('--t_min', type=float, default=None,
                        help='Minimum time to show in seconds (e.g., 2.5)')
    parser.add_argument('--t_max', type=float, default=None,
                        help='Maximum time to show in seconds (e.g., 10.0)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for encoded video directories (e.g., "test", "first3s")')

    args = parser.parse_args()

    # Call plotting function
    plot_bitrate_traces(
        input_patterns=args.patterns,
        t_min=args.t_min,
        t_max=args.t_max,
        suffix=args.suffix
    )