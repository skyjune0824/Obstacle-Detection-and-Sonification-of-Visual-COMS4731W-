
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocess.data_management import load_data
from config import RAW_CSV_PATH


def get_random_h5_path_from_csv(csv_path: Path) -> Path:
    """
    from raw.csv choose 1 random
    """
    if not csv_path.exists():
        print(f"Cannot find: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)

        if 'path' not in df.columns:
            sys.exit(1)

        random_row = df.sample(n=1).iloc[0]
        selected_path = Path(random_row['path'])

        print(f"   File choice: {selected_path.name}")

        return selected_path

    except Exception as e:
        print(f"Error while loading csv: {e}")
        sys.exit(1)


# Old Ransac floor removal
# def filter_floor_with_ransac(points_3d: np.ndarray, distance_threshold: float = 0.05, ransac_n: int = 3) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Use ransac to remove floor from 3d cloud

#     Args:
#         points_3d (np.ndarray): N x 3 3D point cloud.
#         distance_threshold (float): 
#         ransac_n (int): Minimum number of points required to fit the model.

#     Returns:
#         np.ndarray: Point cloud without floor.
#     """

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_3d)

#     plane_model, inliers = pcd.segment_plane(
#         distance_threshold=distance_threshold,
#         ransac_n=ransac_n,
#         num_iterations=1000
#     )

#     if len(inliers) < 10:  # Floor not found or too small floor area
#         print("Can't find floor object")
#         return points_3d, np.arange(len(points_3d))

#     all_indices = np.arange(len(points_3d))
#     outlier_mask = np.ones(len(points_3d), dtype=bool)
#     outlier_mask[inliers] = False

#     non_floor_indices = all_indices[outlier_mask]
#     non_floor_points = points_3d[non_floor_indices]

#     return non_floor_points, non_floor_indices


# Camera info from NYC V2
FX_D = 5.1885790117450188e+02
FY_D = 5.1946961112127485e+02
CX_D = 3.2558244941119034e+02
CY_D = 2.5373616633400465e+02

# Mapping config
ANGLE_RESOLUTION = 2.0
Y_FLOOR_LIMIT = -0.5
Y_CEILING_LIMIT = 1.0

# Pitch filtering config
PITCH_THRESHOLD = 0.2


def visualize_depth_map(rgb_image: np.ndarray, depth_array: np.ndarray, title: str = "2D Depth Map Visualization"):
    """
    Create a depth map, BEV
    """
    
    H, W = depth_array.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth_array.astype(np.float32)
    X = (u - CX_D) * Z / FX_D
    Y = (v - CY_D) * Z / FY_D

    mask = (Z > 0.1) & (Z < 10.0)

    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]

    u_valid = u[mask]
    v_valid = v[mask]

    if len(Z_valid) < 100:
        print("Not enough points for filtering.")
        return

    stable_z_mask = Z_valid > 0.5 
    
    Z_filter = Z_valid[stable_z_mask]
    Y_filter = Y_valid[stable_z_mask]
    
    pitch_ratios = Y_filter / Z_filter
    median_ratio = np.median(pitch_ratios)
    
    is_floor_mask_stat = np.abs(pitch_ratios - median_ratio) < PITCH_THRESHOLD
    
    original_valid_indices = np.where(stable_z_mask)[0]
    
    non_floor_indices = original_valid_indices[~is_floor_mask_stat]
    
    u_ransac_filtered = u_valid[non_floor_indices]
    v_ransac_filtered = v_valid[non_floor_indices]

    X_valid = X_valid[non_floor_indices]
    Y_valid = Y_valid[non_floor_indices]
    Z_valid = Z_valid[non_floor_indices]

    R = np.sqrt(X_valid**2 + Z_valid**2)
    Theta_rad = np.arctan2(X_valid, Z_valid)

    Theta_deg = np.rad2deg(Theta_rad)
    Theta_deg_normalized = np.where(Theta_deg < 0, Theta_deg + 360, Theta_deg)

    num_bins = int(360 / ANGLE_RESOLUTION)
    bins = np.linspace(0, 360, num_bins + 1)
    angle_indices = np.digitize(Theta_deg_normalized, bins) - 1

    filtered_indices = []

    for i in range(num_bins):
        bin_mask = (angle_indices == i)

        if np.any(bin_mask):
            r_in_bin = R[bin_mask]

            min_r_relative_index = np.argmin(r_in_bin)

            original_indices_in_bin = np.where(bin_mask)[0]

            filtered_indices.append(
                original_indices_in_bin[min_r_relative_index])

    filtered_indices = np.array(filtered_indices)

    X_filtered = X_valid[filtered_indices]
    Z_filtered = Z_valid[filtered_indices]
    R_filtered = R[filtered_indices]
    
    u_final_filtered = u_ransac_filtered[filtered_indices]
    v_final_filtered = v_ransac_filtered[filtered_indices]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=16)

    ax1 = axes[0]

    if rgb_image.shape[0] == 3:
        display_rgb = np.transpose(rgb_image, (1, 2, 0))
    else:
        display_rgb = rgb_image

    ax1.imshow(display_rgb)
    ax1.set_title('1. Original RGB Image (Front View)')
    ax1.axis('off')

    ax1.scatter(u_final_filtered, v_final_filtered,
                c=R_filtered,
                cmap='jet',
                s=50,
                edgecolors='white',
                linewidths=1.5,
                marker='o',
                )

    ax1.legend(loc='lower left', framealpha=0.8)

    ax2 = axes[1]
    scatter = ax2.scatter(X_filtered, Z_filtered,
                          c=R_filtered,
                          cmap='jet',
                          s=50,
                          marker='o'
                          )

    plt.colorbar(scatter, ax=ax2, label='Radial Distance (Meters)')

    ax2.scatter(0, 0, color='red', marker='^', s=200, label='Camera Position')

    ax2.set_title("2. Filtered Bird's Eye View (Nearest Obstacle)")
    ax2.set_xlabel('Lateral Distance (X-axis, Meters)')
    ax2.set_ylabel('Forward Distance (Z-axis, Meters)')

    ax2.set_xlim([-4, 4])
    ax2.set_ylim([0, 8])
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


if __name__ == "__main__":

    SAMPLE_H5_PATH = get_random_h5_path_from_csv(RAW_CSV_PATH)

    if not SAMPLE_H5_PATH.exists():
        print(f"Cannot find chosen h5 file: {SAMPLE_H5_PATH}")
        sys.exit(1)

    loaded_data = load_data(SAMPLE_H5_PATH)

    if loaded_data:
        rgb_image, depth_map = loaded_data

        visualize_depth_map(rgb_image, depth_map, title=f"Depth Map from {SAMPLE_H5_PATH.name} (Random Sample)")