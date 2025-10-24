import kagglehub
from pathlib import Path

KAGGLE_DATASET_HANDLE = "artemmmtry/nyu-depth-v2" # currently not working
# try manually download from: https://www.kaggle.com/datasets/artemmmtry/nyu-depth-v2

def download_dataset(handle: str = KAGGLE_DATASET_HANDLE, force: bool = False):
    '''
    Download dataset from KaggleHub.
    
    Args:
        handle (str): The dataset handle (e.g., 'user/dataset-name').
        force (bool): If True, forces the download to overwrite existing files.
    '''
    
    print(f"Downloading '{handle}' (Force overwrite: {force})...")

    current_script_dir = Path(__file__).resolve().parent
    download_dir = current_script_dir.parent / "data"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = str(download_dir)

    try:
        local_path = kagglehub.dataset_download(handle, path=target_path, force_download=force)

        print(f"Download finished.")
        print(f"    Path: {local_path}")

        return local_path

    except Exception as e:
        print(f"Download Error Occured: {e}")
        return None
