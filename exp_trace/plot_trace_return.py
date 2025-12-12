#!/usr/bin/env python3
"""
plot_trace_return.py - Plot discounted reward returns from experimental traces

This script reads experimental trace data from exp_trace directories,
computes rewards using configuration files, and plots discounted returns
with optional time-based filtering.

Usage:
    python plot_trace_return.py \
        --trace-dir exp_trace/12_10_v3 \
        --config net_util/net_config/12_10_v4/STA1_STA2_fixed.json \
        --maxtime 1765358932.1074548 \
        --gamma 0.99 \
        --output returns_plot.png
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# Try to import pandas, but don't fail if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_trace.plot_utils import (
    create_figure,
    apply_scientific_style,
    save_figure,
    PlotTheme,
)
from util.trace_collec import trace_filter, flatten_leaves


def discover_trials(trace_dir: str) -> List[Tuple[str, str, float]]:
    """
    Discover trial directories and extract their information.

    Args:
        trace_dir: Path to trace directory containing trial folders

    Returns:
        List of (trial_path, trial_name, start_timestamp) tuples sorted by timestamp
    """
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace directory does not exist: {trace_dir}")

    # Pattern to match: IL_<interference>_trial_<YYYYMMDD-HHMMSS>
    pattern = re.compile(r'IL_(\d+)_trial_(\d{8})-(\d{6})')
    trials = []

    for trial_dir in trace_path.iterdir():
        if not trial_dir.is_dir():
            continue

        match = pattern.match(trial_dir.name)
        if not match:
            continue

        interference = int(match.group(1))
        date_str = match.group(2)
        time_str = match.group(3)

        # Parse timestamp
        try:
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            timestamp = dt.timestamp()
        except ValueError:
            print(f"Warning: Could not parse timestamp from {trial_dir.name}")
            continue

        trials.append((str(trial_dir), trial_dir.name, timestamp))

    # Sort by timestamp
    trials.sort(key=lambda x: x[2])
    return trials


def load_reward_config(config_path: str) -> Dict:
    """
    Load reward configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Reward configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('reward_cfg', {})
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing config file {config_path}: {e}")
        return {}


def get_default_reward_config() -> Dict:
    """
    Get default reward configuration.

    Returns:
        Default reward configuration dictionary
    """
    return {
        "acc_bitrate": {
            "rule": "stat_bitrate",
            "from": "acc_bitrate",
            "args": {"alpha": 2e-07},
            "pos": "flow"
        },
        "diff_bitrate": {
            "rule": "stat_bitrate",
            "from": "diff_bitrate",
            "args": {"alpha": -1e-07},
            "pos": "flow"
        },
        "outage_rate": {
            "rule": "scale_outage",
            "from": "outage_rate",
            "args": {"zeta": -4.0},
            "pos": "flow"
        }
    }


def process_trace_file(filepath: str, reward_cfg: Dict, maxtime: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """
    Process rollout.jsonl to extract rewards and timestamps.

    Args:
        filepath: Path to rollout.jsonl file
        reward_cfg: Reward configuration dictionary
        maxtime: Optional maximum timestamp to filter records

    Returns:
        (timestamps, rewards) tuples filtered by maxtime if provided
    """
    timestamps = []
    rewards = []

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num} in {filepath}")
                    continue

                # Check for timestamp field
                if 't' not in record:
                    continue

                timestamp = record['t']

                # Apply maxtime filter if specified
                if maxtime is not None and timestamp > maxtime:
                    continue

                # Compute reward using trace_filter
                try:
                    filtered = trace_filter(record, reward_cfg)
                    values = flatten_leaves(filtered)

                    # Sum all reward components
                    reward = sum(values) if values else 0.0
                except Exception as e:
                    # Fallback: use 0 reward if filtering fails
                    reward = 0.0

                timestamps.append(timestamp)
                rewards.append(reward)

    except FileNotFoundError:
        print(f"Warning: Trace file not found: {filepath}")
        return [], []

    return timestamps, rewards


def compute_discounted_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Apply exponential discounting to compute returns.

    Args:
        rewards: List of instantaneous rewards
        gamma: Discount factor (default: 0.99)

    Returns:
        List of discounted returns
    """
    if not rewards:
        return []

    # Compute discounted return from each time step to the end
    n = len(rewards)
    returns = [0.0] * n
    returns[-1] = rewards[-1]  # Last step return is just the last reward

    # Work backwards computing the discounted sum of future rewards
    for i in range(n - 2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i + 1]

    return returns


def apply_smoothing(data: List[float], window: int) -> List[float]:
    """
    Apply moving average smoothing using plot_utils.moving_average.

    Args:
        data: Input data series
        window: Window size for smoothing

    Returns:
        Smoothed data series
    """
    from exp_trace.plot_utils import moving_average
    return moving_average(data, window, mode="same")


def filter_outliers(data: List[float], percentile: float = 5.0) -> Tuple[List[float], float, float]:
    """
    Filter outliers using percentile method.

    Args:
        data: Input data series
        percentile: Percentile for outlier filtering (default: 5.0)

    Returns:
        (filtered_data, ymin, ymax) for appropriate y-axis limits
    """
    if not data:
        return data, 0.0, 1.0

    arr = np.array(data)

    # Calculate percentiles
    lower = np.percentile(arr, percentile)
    upper = np.percentile(arr, 100 - percentile)

    # Filter data within percentiles
    mask = (arr >= lower) & (arr <= upper)
    filtered_data = arr[mask].tolist()

    return filtered_data, lower, upper


# --------------------- Cache Management ---------------------

def get_cache_path(output_path: str) -> str:
    """Generate cache path from output path"""
    output = Path(output_path)
    return str(output.with_suffix(output.suffix + ".csv"))


def load_cache(cache_path: str) -> Dict[str, Dict]:
    """Load cached reward data from CSV"""
    cache_data = {}

    print(cache_path)
    if not Path(cache_path).exists():
        return cache_data
    print(cache_path)
    try:
        if HAS_PANDAS:
            # Use pandas for faster loading
            df = pd.read_csv(cache_path)
            for trial_name in df['trial_name'].unique():
                trial_df = df[df['trial_name'] == trial_name]
                cache_data[trial_name] = {
                    'timestamps': trial_df['timestamp'].tolist(),
                    'rewards': trial_df['reward'].tolist(),
                    'returns': trial_df['return'].tolist(),
                    'file_mtime': trial_df['file_mtime'].iloc[0] if 'file_mtime' in trial_df.columns else 0
                }
        else:
            # Use standard library
            with open(cache_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trial_name = row['trial_name']
                    if trial_name not in cache_data:
                        cache_data[trial_name] = {
                            'timestamps': [],
                            'rewards': [],
                            'returns': [],
                            'file_mtime': float(row.get('file_mtime', 0))
                        }
                    cache_data[trial_name]['timestamps'].append(float(row['timestamp']))
                    cache_data[trial_name]['rewards'].append(float(row['reward']))
                    cache_data[trial_name]['returns'].append(float(row['return']))
    except Exception as e:
        print(f"Warning: Failed to load cache {cache_path}: {e}")
        return {}

    return cache_data


def save_cache(cache_path: str, trials_data: List[Tuple], trial_paths: List[str]):
    """Save trial data to CSV cache"""
    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if HAS_PANDAS:
            # Use pandas for faster saving
            rows = []
            for (trial_name, timestamps, rewards, returns), trial_path in zip(trials_data, trial_paths):
                file_mtime = Path(trial_path).stat().st_mtime if trial_path else 0
                for i, (ts, r, ret) in enumerate(zip(timestamps, rewards, returns)):
                    rows.append({
                        'trial_name': trial_name,
                        'step': i,
                        'timestamp': ts,
                        'reward': r,
                        'return': ret,
                        'file_mtime': file_mtime
                    })

            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(cache_path, index=False)
        else:
            # Use standard library
            with open(cache_path, 'w', newline='') as f:
                fieldnames = ['trial_name', 'step', 'timestamp', 'reward', 'return', 'file_mtime']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for (trial_name, timestamps, rewards, returns), trial_path in zip(trials_data, trial_paths):
                    file_mtime = Path(trial_path).stat().st_mtime if trial_path else 0
                    for i, (ts, r, ret) in enumerate(zip(timestamps, rewards, returns)):
                        writer.writerow({
                            'trial_name': trial_name,
                            'step': i,
                            'timestamp': ts,
                            'reward': r,
                            'return': ret,
                            'file_mtime': file_mtime
                        })
    except Exception as e:
        print(f"Warning: Failed to save cache {cache_path}: {e}")


def is_cache_fresh(cache_path: str, trial_paths: List[str]) -> bool:
    """Check if cache is newer than trace files"""
    if not Path(cache_path).exists():
        return False

    # try:
    #     cache_mtime = Path(cache_path).stat().st_mtime

    #     for trial_path in trial_paths:
    #         if Path(trial_path).exists() and Path(trial_path).stat().st_mtime > cache_mtime:
    #             return False

    #     return True
    # except Exception:
    #     return False
    return True


def plot_returns(trials_data: List[Tuple[str, List[float], List[float], List[float]]],
                gamma: float,
                title: str = "Reward Returns",
                output_path: str = "returns_plot.png",
                return_type: str = "discounted",
                maxticks: Optional[float] = None,
                no_time_axis: bool = False,
                smooth: bool = False,
                smooth_window: int = 100,
                ignore_outliers: bool = False,
                outlier_percentile: float = 5.0):
    """
    Plot discounted returns for multiple trials as a single continuous timeline.

    Args:
        trials_data: List of (trial_name, timestamps, rewards, returns) tuples
        gamma: Discount factor used
        title: Plot title
        output_path: Output file path for the plot
        return_type: Type of returns plotted ("discounted" or "cumulative")
        maxticks: Optional maximum x-axis tick value to remap current x values
        no_time_axis: If True, use step numbers instead of time values
        smooth: If True, apply smoothing and show raw data in background
        smooth_window: Window size for moving average smoothing
        ignore_outliers: If True, filter outliers from plot
        outlier_percentile: Percentile for outlier filtering
    """
    # Create figure
    fig, ax = create_figure(size="large")

    # Combine all trials into a single timeline
    all_x_values = []
    all_returns = []

    # Sort trials by their start time
    trials_data.sort(key=lambda x: x[1][0] if x[1] else float('inf'))

    for trial_idx, (trial_name, timestamps, _, returns) in enumerate(trials_data):
        if not timestamps:
            print(f"Warning: No data for trial {trial_name}")
            continue

        if no_time_axis:
            # Use step numbers instead of time
            x_values = list(range(len(all_returns), len(all_returns) + len(returns)))
        else:
            # Convert timestamps to relative time in minutes from the very first trial
            if all_x_values:
                # Adjust timestamps relative to the first trial's start time
                first_timestamp = trials_data[0][1][0] if trials_data[0][1] else 0
                x_values = [(t - first_timestamp) / 60.0 for t in timestamps]
            else:
                x_values = [(t - timestamps[0]) / 60.0 for t in timestamps]

        # Add to combined dataset
        all_x_values.extend(x_values)
        all_returns.extend(returns)

    # Plot combined data as a single line
    if all_x_values and all_returns:
        # Sort by x values to ensure proper line drawing
        combined_data = sorted(zip(all_x_values, all_returns))
        sorted_x, sorted_returns = zip(*combined_data)
        sorted_x = list(sorted_x)
        sorted_returns = list(sorted_returns)

        # Apply maxticks scaling if specified
        if maxticks is not None and len(sorted_x) > 0:
            x_min, x_max = min(sorted_x), max(sorted_x)
            if x_max > x_min:  # Avoid division by zero
                scale_factor = maxticks / (x_max - x_min)
                sorted_x = [(x - x_min) * scale_factor for x in sorted_x]

        # Downsample for better visualization if too many points
        if len(sorted_x) > 10000:
            stride = max(1, len(sorted_x) // 5000)
            # stride = 1
            sorted_x = sorted_x[::stride]
            sorted_returns = sorted_returns[::stride]

        # Handle smoothing if enabled
        if smooth:
            # Plot raw data in background first
            ax.plot(sorted_x, sorted_returns,
                    color='gray',
                    linewidth=1,
                    alpha=0.3,
                    label=f'Raw Data ({len(trials_data)} trials)')

            # Apply smoothing
            smoothed_returns = apply_smoothing(sorted_returns, smooth_window)
            ax.plot(sorted_x, smoothed_returns,
                    color=PlotTheme.COLORS['primary0'],
                    linewidth=2,
                    alpha=0.9,
                    label=f'Smoothed {return_type.capitalize()} Returns (window={smooth_window})')
        else:
            # Regular plot without smoothing
            ax.plot(sorted_x, sorted_returns,
                    color=PlotTheme.COLORS['primary0'],
                    linewidth=2,
                    alpha=0.9,
                    label=f'{return_type.capitalize()} Returns ({len(trials_data)} trials)')

        # Handle outlier filtering for y-axis limits
        if ignore_outliers:
            filtered_returns, ymin, ymax = filter_outliers(sorted_returns, outlier_percentile)
            if filtered_returns:  # Only set if we have data after filtering
                # Add some padding to the limits
                y_range = ymax - ymin
                if y_range > 0:
                    ax.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)
                else:
                    # If all values are the same, add a small symmetric range
                    center = (ymin + ymax) / 2
                    ax.set_ylim(center - 1, center + 1)

    # Apply styling
    apply_scientific_style(ax)

    # Set labels and title
    if no_time_axis:
        ax.set_xlabel("Steps", fontsize=12)
    else:
        ax.set_xlabel("Time (minutes)", fontsize=12)

    if return_type == "cumulative":
        ax.set_ylabel("Cumulative Reward", fontsize=12)
        ax.set_title(f"{title} (Cumulative)", fontsize=14)
    else:
        ax.set_ylabel("Discounted Return", fontsize=12)
        ax.set_title(f"{title} (γ={gamma})", fontsize=14)

    if len(trials_data) > 1:
        ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3)

    # Save figure
    save_figure(fig, output_path)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot discounted reward returns from experimental traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all trials with default config
  python plot_trace_return.py --trace-dir exp_trace/12_10_v3

  # Plot with specific config and time limit
  python plot_trace_return.py \\
      --trace-dir exp_trace/12_10_v3 \\
      --config net_util/net_config/12_10_v4/STA1_STA2_fixed.json \\
      --maxtime 1765358932.1074548 \\
      --gamma 0.99
        """
    )

    parser.add_argument(
        "--trace-dir",
        type=str,
        default="exp_trace/12_10_v3",
        help="Path to trace directory (default: exp_trace/12_10_v3)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file containing reward_cfg"
    )

    parser.add_argument(
        "--maxtime",
        type=float,
        default=None,
        help="Maximum Unix timestamp to filter records (e.g., 1765358932.1074548)"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discount factor for returns (default: 0.999)"
    )

    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Plot cumulative sum of rewards instead of discounted returns"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="returns_plot.png",
        help="Output plot file path (default: returns_plot.png)"
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Discounted Reward Returns",
        help="Plot title (default: Discounted Reward Returns)"
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum number of trials to trials (useful for testing)"
    )

    parser.add_argument(
        "--maxticks",
        type=float,
        default=None,
        help="Maximum x-axis tick value (maps current x to this max value)"
    )

    parser.add_argument(
        "--no-time-axis",
        action="store_true",
        help="Remove time axis and use step numbers instead"
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth mode with background raw data"
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Window size for moving average smoothing (default: 100)"
    )

    parser.add_argument(
        "--ignore-outliers",
        action="store_true",
        help="Remove extreme values from plot"
    )

    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=5.0,
        help="Percentile for outlier filtering (default: 5.0)"
    )

    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only update cache, no plotting"
    )

    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore cache and reprocess all trials"
    )

    args = parser.parse_args()

    try:
        # Load reward configuration
        if args.config:
            reward_cfg = load_reward_config(args.config)
            if not reward_cfg:
                print("Warning: No reward_cfg found in config file, using defaults")
                reward_cfg = get_default_reward_config()
        else:
            print("No config file specified, using default reward configuration")
            reward_cfg = get_default_reward_config()

        # Discover trials
        print(f"Discovering trials in {args.trace_dir}...")
        trials = discover_trials(args.trace_dir)
        print(f"Found {len(trials)} trials")

        # Limit number of trials if specified
        if args.max_trials is not None:
            trials = trials[:args.max_trials]
            print(f"Processing first {len(trials)} trials")

        if not trials:
            print("No valid trials found!")
            sys.exit(1)

        # Setup cache
        cache_path = get_cache_path(args.output)
        trial_paths = [str(Path(trial_path) / "rollout.jsonl") for trial_path, _, _ in trials]

        # Check if we should use cache
        use_cache = not args.force_reprocess and is_cache_fresh(cache_path, trial_paths)

        if use_cache and not args.force_reprocess:
            print(f"\nLoading from cache: {cache_path}")
            cache_data = load_cache(cache_path)
            trials_data = []
            trial_names_from_cache = set(cache_data.keys())

            # Load cached data
            for trial_path, trial_name, _ in trials:
                if trial_name in cache_data:
                    cached = cache_data[trial_name]
                    trials_data.append((trial_name, cached['timestamps'], cached['rewards'], cached['returns']))

            # Find new trials to process
            new_trials = [(t, n, ts) for t, n, ts in trials if n not in trial_names_from_cache]

            if new_trials:
                print(f"\nProcessing {len(new_trials)} new trials...")
        else:
            if args.force_reprocess:
                print("\nForce reprocess: ignoring cache")
            else:
                print(f"\nCache not found or stale: {cache_path}")
            new_trials = trials
            trials_data = []

        # Process new trials
        return_type = "cumulative" if args.cumulative else "discounted"
        for trial_path, trial_name, _ in new_trials:
            print(f"\nProcessing trial: {trial_name}")
            rollout_file = Path(trial_path) / "rollout.jsonl"

            if not rollout_file.exists():
                print(f"  Warning: rollout.jsonl not found")
                continue

            # Extract data
            timestamps, rewards = process_trace_file(str(rollout_file), reward_cfg, args.maxtime)

            if not timestamps:
                print(f"  Warning: No valid data extracted")
                continue

            # Compute returns
            if args.cumulative:
                # Use cumulative sum instead of discounted returns
                returns = np.cumsum(rewards).tolist()
            else:
                # Use discounted returns
                returns = compute_discounted_returns(rewards, args.gamma)

            print(f"  Extracted {len(timestamps)} records")
            print(f"  Total reward: {sum(rewards):.2f}")

            trials_data.append((trial_name, timestamps, rewards, returns))

        # Sort trials by timestamp for consistent ordering
        trials_data.sort(key=lambda x: x[1][0] if x[1] else float('inf'))

        # Save cache if we have new data
        if new_trials and trials_data:
            print(f"\nUpdating cache: {cache_path}")
            save_cache(cache_path, trials_data, trial_paths)

        if not trials_data:
            print("\nNo data to plot!")
            sys.exit(1)

        # Create plot (unless cache-only mode)
        if not args.cache_only:
            print(f"\nCreating plot...")
            plot_returns(trials_data, args.gamma, args.title, args.output, return_type,
                        args.maxticks, args.no_time_axis, args.smooth, args.smooth_window,
                        args.ignore_outliers, args.outlier_percentile)

            # Print summary
            print(f"\n✅ Plot complete!")
            print(f"  Processed {len(trials_data)} trials with data")
            print(f"  Discount factor: {args.gamma}")
            print(f"  Output: {args.output}")
            if args.smooth:
                print(f"  Smoothing: enabled (window={args.smooth_window})")
            if args.ignore_outliers:
                print(f"  Outlier filtering: enabled (percentile={args.outlier_percentile})")
        else:
            print(f"\n✅ Cache updated!")
            print(f"  Processed {len(trials_data)} trials with data")
            print(f"  Cache: {cache_path}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()