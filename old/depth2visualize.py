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
from config import RAW_CSV_PATH, DEBUG


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


# # Camera info from NYC V2
# FX_D = 5.1885790117450188e+02
# FY_D = 5.1946961112127485e+02
# CX_D = 3.2558244941119034e+02
# CY_D = 2.5373616633400465e+02

# # Mapping config
# Y_FLOOR_LIMIT = -0.5
# Y_CEILING_LIMIT = 1.0

# # Pitch filtering config
# PITCH_THRESHOLD = 0.2

ANGLE_RESOLUTION = 1.0


def depth2ad(depth_array: np.ndarray, title: str = "Filtered Bearings and Ranges (BEV)") -> np.ndarray:
    """
    Using Relative Depth Map to calculate relative distance.

    Args:
        depth_array (np.ndarray): 0-255 array of depth map
        title (str): name of visualization
        
    Returns:
        np.ndarray: list of (angle, distance) pairs.
    """
    
    H, W = depth_array.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth_array.astype(np.float32)

    F_REL = W
    CX_REL = W / 2.0
    
    X = (u - CX_REL) * Z / F_REL
    
    mask = (Z > 0.1)

    X_valid = X[mask]
    Z_valid = Z[mask]
    
    u_valid = u[mask]
    v_valid = v[mask]

    if len(Z_valid) < 100:
        if DEBUG:
            print("Not enough points for filtering.")
        return np.empty((0, 2))

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
            r_in_bin = 255 - r_in_bin

            min_r_relative_index = np.argmin(r_in_bin)

            original_indices_in_bin = np.where(bin_mask)[0]

            filtered_indices.append(
                original_indices_in_bin[min_r_relative_index])

    filtered_indices = np.array(filtered_indices, dtype=np.int64)

    X_filtered = X_valid[filtered_indices]
    Z_filtered = Z_valid[filtered_indices]
    R_filtered = R[filtered_indices]
    Theta_filtered = Theta_deg_normalized[filtered_indices]

    u_final_filtered = u_valid[filtered_indices]
    v_final_filtered = v_valid[filtered_indices]
    
    result_array = np.vstack((
    Theta_filtered, 
    R_filtered,
    u_final_filtered,
    v_final_filtered
    )).T
    
    if DEBUG:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(title, fontsize=16)

        ax1 = axes[0]
        
        depth_display = ax1.imshow(depth_array)
        ax1.set_title("1. Depth Map (Front View) with Filtered Points")
        
        scatter1 = ax1.scatter(u_final_filtered, v_final_filtered,
                            c=R_filtered,
                            cmap='jet',
                            s=30,
                            edgecolors='white',
                            marker='o',
                            )
        
        ax1.axis('off') 
        ax1.legend(loc='lower left', framealpha=0.8)

        ax2 = axes[1]
        
        scatter2 = ax2.scatter(X_filtered, 255-Z_filtered,
                              c=R_filtered,
                              cmap='jet',
                              s=30,
                              marker='o'
                              )

        plt.colorbar(scatter2, ax=ax2, label='Relative Radial Distance')

        ax2.scatter(0, 0, color='red', marker='^', s=200, label='Camera Position')

        ax2.set_title("2. Filtered Bird's Eye View (Nearest Obstacle)")
        ax2.set_xlabel('Relative Lateral Distance (X-axis)')
        ax2.set_ylabel('Relative Forward Distance (Z-axis)')

        max_abs_x = np.max(np.abs(X_filtered)) if len(X_filtered) > 0 else 1
        max_z = np.max(Z_filtered) if len(Z_filtered) > 0 else 1
        max_coord = max(max_abs_x * 1.5, max_z * 1.2, 5)
        
        ax2.set_xlim([-max_coord * 0.5, max_coord * 0.5]) 
        ax2.set_ylim([0, max_coord])
        
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.show()

    return result_array


if __name__ == "__main__":

    SAMPLE_H5_PATH = get_random_h5_path_from_csv(RAW_CSV_PATH)

    if not SAMPLE_H5_PATH.exists():
        print(f"Cannot find chosen h5 file: {SAMPLE_H5_PATH}")
        sys.exit(1)

    loaded_data = load_data(SAMPLE_H5_PATH)

    if loaded_data:
        rgb_image, depth_map = loaded_data
        
        result_array = depth2ad(depth_map, debug = True)

    # if loaded_data:
    #     rgb_image, depth_map = loaded_data
        
    #     if rgb_image.ndim == 3 and rgb_image.shape[0] in [3, 4]:
    #         display_rgb = np.transpose(rgb_image, (1, 2, 0))
    #     else:
    #         display_rgb = rgb_image
        
    #     result_array = depth2ad(depth_map, debug = False)

    #     R_filtered = result_array[:, 1]
    #     u_filtered = result_array[:, 2]
    #     v_filtered = result_array[:, 3]

    #     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
    #     ax1 = axes[0]
    #     ax1.imshow(display_rgb)
    #     ax1.set_title(f"1. RGB Image with Filtered Points from {SAMPLE_H5_PATH.name}")
    #     ax1.axis('off')

    #     scatter = ax1.scatter(u_filtered, v_filtered,
    #                         c=R_filtered,
    #                         cmap='jet',
    #                         s=50,
    #                         edgecolors='white',
    #                         linewidths=1.5,
    #                         marker='o',
    #                         )

    #     fig.colorbar(scatter, ax=ax1, label='Relative Radial Distance', orientation='vertical', fraction=0.046, pad=0.04)
    #     ax1.legend(loc='lower left', framealpha=0.8)

    #     X_filtered = (u_filtered - (depth_map.shape[1] / 2.0)) * R_filtered / depth_map.shape[1]
    #     Z_filtered = np.sqrt(R_filtered**2 - X_filtered**2) 

    #     ax2 = axes[1]
        
    #     scatter_bev = ax2.scatter(X_filtered, Z_filtered,
    #                               c=R_filtered,
    #                               cmap='jet',
    #                               s=50,
    #                               marker='o'
    #                               )

    #     ax2.scatter(0, 0, color='red', marker='^', s=200, label='Camera Position')
        
    #     ax2.set_title("2. Filtered Bird's Eye View (Nearest Obstacle)")
    #     ax2.set_xlabel('Relative Lateral Distance (X-axis)')
    #     ax2.set_ylabel('Relative Forward Distance (Z-axis)')
        
    #     max_abs_x = np.max(np.abs(X_filtered)) if len(X_filtered) > 0 else 1
    #     max_z = np.max(Z_filtered) if len(Z_filtered) > 0 else 1
    #     max_coord = max(max_abs_x * 1.5, max_z * 1.2, 5) 
        
    #     ax2.set_xlim([-max_coord * 0.5, max_coord * 0.5]) 
    #     ax2.set_ylim([0, max_coord])
        
    #     ax2.set_aspect('equal', adjustable='box')
    #     ax2.grid(True, linestyle='--', alpha=0.5)
    #     ax2.legend(loc='upper right')
        
    #     plt.tight_layout()
    #     plt.show()