import kagglehub
import h5py
import numpy as np
from pathlib import Path
from config import DOWNLOAD_DIR, KAGGLE_DATASET_HANDLE, RGB_KEY, DEPTH_KEY

# currently not working
# try manually download from: https://www.kaggle.com/datasets/artemmmtry/nyu-depth-v2

def download_dataset(handle: str = KAGGLE_DATASET_HANDLE, force: bool = False):
    '''
    Download dataset from KaggleHub.

    Args:
        handle (str): The dataset handle (e.g., 'user/dataset-name').
        force (bool): If True, forces the download to overwrite existing files.
    '''

    print(f"Downloading '{handle}' (Force overwrite: {force})...")

    target_path = str(DOWNLOAD_DIR)

    try:
        local_path = kagglehub.dataset_download(
            handle, path=target_path, force_download=force)

        print(f"Download finished.")
        print(f"    Path: {local_path}")

        return local_path

    except Exception as e:
        print(f"Download Error Occured: {e}")
        return None


def load_data(h5_file_path: Path):
    """
    Draw RGB image and Depth map from single h5 file

    Args:
        h5_file_path (Path): path of h5 file.

    Returns:
        tuple: (rgb_image: np.ndarray, depth_map: np.ndarray)
    """

    try:
        rgb_image = None
        depth_map = None

        with h5py.File(h5_file_path, 'r') as f:
            if RGB_KEY in f and DEPTH_KEY in f:
                rgb_image = np.array(f[RGB_KEY])
                depth_map = np.array(f[DEPTH_KEY])
            else:
                print(
                    f"{h5_file_path} does not contain expected keys 'rgb' and 'depth'.")
                return None

    except Exception as e:
        print(f"Error occured during opening file {h5_file_path}: {e}")
        return None

    return rgb_image, depth_map
