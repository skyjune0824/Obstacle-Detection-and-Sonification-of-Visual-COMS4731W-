# Load Module
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.MDE.pipeline import MDE_Pipeline
import matplotlib.pyplot as plt


def plot_audio(data):
    """ Plot Audio

    Creates two plots conveying frequency vs closest obstacle distance
    and volume vs closest obstacle distance for the three spatial zones.
    """

    # Spatial Zones and Respective Colors
    zones = ["right"] #  "left", "right"
    colors = {"left": "red", "center": "green", "right": "blue"}

    # Freq + Volume Plots
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for zone in zones:
        # Capture distance from tuples
        distances = [x[0] for x in data[zone]]
        # Capture frequency from tuples
        frequencies = [x[1] for x in data[zone]] 
        axes[0].plot(distances, frequencies, marker='o', linestyle='-', color=colors[zone], label=zone)
        print(f"ZONE {zone}: {frequencies}, {distances}")

    # Plot Syntax
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Distance vs Frequency for each Zone")
    axes[0].legend()
    axes[0].grid(True)

    # Distance vs Volume
    for zone in zones:
        # Capture distance from tuples
        distances = [x[0] for x in data[zone]]
        # Capture frequency from tuples
        volumes = [x[2] for x in data[zone]]
        axes[1].plot(distances, volumes, marker='o', linestyle='-', color=colors[zone], label=zone)

    # Plot Syntax
    axes[1].set_xlabel("Min Distance to Obstacle")
    axes[1].set_ylabel("Volume")
    axes[1].set_title("Distance vs Volume for each Zone")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test Samples: Insert Your own Sample Paths Here.
    # Example One
    MDE = MDE_Pipeline(rate=30, threshold=5)
    plotting_data = MDE.pipeline("test/data/Personal/kitti_00.avi")
    plot_audio(plotting_data)

    # Example Two
    MDE = MDE_Pipeline(rate=30, threshold=5)
    plotting_data = MDE.pipeline("test/data/Personal/test.MOV")
    plot_audio(plotting_data)