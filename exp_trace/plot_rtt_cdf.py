#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_rtts(file_path: Path):
    """Read RTT values (2nd column) from a text file."""
    rtts = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    rtts.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(rtts)


def plot_cdf(rtts: np.ndarray, label: str):
    """Plot CDF for a given RTT array."""
    rtts_sorted = np.sort(rtts)
    cdf = np.arange(1, len(rtts_sorted) + 1) / len(rtts_sorted)
    plt.plot(rtts_sorted * 1000, cdf, label=label)  # convert to ms


def main():
    parser = argparse.ArgumentParser(description="Plot RTT CDFs from log files.")
    parser.add_argument("log_files", nargs="+", help="One or more RTT log text files")
    args = parser.parse_args()

    plt.figure(figsize=(7, 5))
    for log_file in args.log_files:
        path = Path(log_file)
        rtts = read_rtts(path)
        if len(rtts) == 0:
            print(f"[WARN] No valid RTTs found in {log_file}")
            continue
        plot_cdf(rtts, label=path.parent.stem)

    plt.xlabel("RTT (ms)")
    plt.ylabel("CDF")
    plt.title("RTT Cumulative Distribution")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Log Files")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
