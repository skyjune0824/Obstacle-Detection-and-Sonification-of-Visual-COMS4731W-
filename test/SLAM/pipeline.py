# Load Module
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from src.SLAM.pipeline import SLAM_Pipeline


def plot_audio(data):
    """Plot Audio per Zone side by side
    """

    zones = ["left", "center", "right"]
    colors = {"left": "red", "center": "green", "right": "blue"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    fig.suptitle("Spatial Audio Mapping by Zone", fontsize=16)

    for col, zone in enumerate(zones):
        if zone not in data or len(data[zone]) == 0:
            continue

        density = np.array([x[0] for x in data[zone]])
        frequencies = np.array([x[1] for x in data[zone]])
        volumes = np.array([x[2] for x in data[zone]])

        # Sort by density
        idx = np.argsort(density)
        density = density[idx]
        frequencies = frequencies[idx]
        volumes = volumes[idx]

        axes[0, col].scatter(density, frequencies, color=colors[zone])
        axes[0, col].set_title(f"{zone.capitalize()} – Frequency")
        axes[0, col].set_ylabel("Frequency (Hz)")
        axes[0, col].grid(True)

        axes[1, col].scatter(density, volumes, color=colors[zone])
        axes[1, col].set_title(f"{zone.capitalize()} – Volume")
        axes[1, col].set_xlabel("Point Density")
        axes[1, col].set_ylabel("Volume")
        axes[1, col].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_mse_over_time(mse_per_frame):
    mse = np.asarray(mse_per_frame)
    mean_mse = mse.mean()

    plt.figure()
    plt.plot(mse)
    plt.axhline(mean_mse, linestyle='--', label=f"Mean MSE = {mean_mse:.3f} px")
    plt.xlabel("Frame")
    plt.ylabel("Mean Squared Error (pixels²)")
    plt.title("Reprojection MSE over time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Test Sample: Insert Your Own Path, and Intrisnics Here
    SFM = SLAM_Pipeline(K=np.array([[780, 0, 540], [0, 780, 960], [0, 0, 1]], dtype=np.float64), cam_speed=1.4, local=True, glob_mem=200)
    audio_trace, est_poses, mse = SFM.pipeline("test/data/Personal/test4.MOV", video=True)

    # Plotting
    plot_audio(audio_trace)
    plot_mse_over_time(mse)

    # KITTI TEST SAMPLE
    SFM = SLAM_Pipeline(K=np.array([[9.842439e+02, 0, 6.900000e+02], [0, 9.808141e+02, 2.331966e+02], [0, 0, 1]]), cam_speed=14, local=True, glob_mem=100)
    audio_trace, est_poses, mse = SFM.pipeline("test/data/Kitti/sequences/00/image_0", video=False)

    # Plotting
    plot_audio(audio_trace)
    plot_mse_over_time(mse)