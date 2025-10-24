from pathlib import Path

#################### PATHS ####################
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "data"
RAW_DIR = DOWNLOAD_DIR / "raw"
PROCESSED_DIR = DOWNLOAD_DIR / "processed"

TRAIN_SPLIT_DIR = "raw/train" 
VAL_SPLIT_DIR = "raw/val"

################ API KEY & URL ################
KAGGLE_DATASET_HANDLE = "artemmmtry/nyu-depth-v2"  # currently not working try: https://www.kaggle.com/datasets/artemmmtry/nyu-depth-v2

##################### KEY #####################
RGB_KEY = 'rgb'
DEPTH_KEY = 'depth'
