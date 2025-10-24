import kagglehub
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path
from config import DOWNLOAD_DIR, KAGGLE_DATASET_HANDLE, RGB_KEY, DEPTH_KEY, TRAIN_SPLIT_DIR, VAL_SPLIT_DIR, RAW_CSV_PATH

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


def save_raw_h5_file_paths(csv_path: Path = RAW_CSV_PATH, force_rebuild: bool = False):
    """
    Create csv files which has all h5 file paths

    Args:
        raw_path: The root directory Path where the h5 files are stored.
        force_rebuild: If True, forces the recreation of the CSV file even if it exists.

    Returns:
        A path to the created CSV file.
    """

    if not force_rebuild and csv_path.exists():
        print(f"\nCSV file already exists at {csv_path}. Loading paths from CSV...")

        return csv_path

    train_root = TRAIN_SPLIT_DIR
    val_root = VAL_SPLIT_DIR

    print(f"\nSearching for training .h5 files recursively under: {train_root}")
    print(f"Searching for validation .h5 files recursively under: {val_root}")

    train_paths: List[Path] = list(train_root.glob('**/*.h5'))
    val_paths: List[Path] = list(val_root.glob('**/*.h5'))

    train_data = [{'split': 'train', 'path': str(p.resolve())} for p in train_paths]
    val_data = [{'split': 'val', 'path': str(p.resolve())} for p in val_paths]

    print(f"\nFound {len(train_paths)} training .h5 files.")
    print(f"Found {len(val_paths)} validation .h5 files.")

    df = pd.DataFrame(train_data + val_data)
    df.to_csv(csv_path, index=False)

    print(f"\nSuccessfully saved all {len(df)} paths to CSV at: {csv_path}")

    return csv_path
