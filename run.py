import sys
import os
import argparse
from src.preprocess.data_management import download_dataset, save_raw_h5_file_paths
from src.pipeline import pipeline
from config import RAW_CSV_PATH, DOWNLOAD_DIR

def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(
        description="Sonification of Visual Obstacle Detection",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # -d, --download: download data
    parser.add_argument(
        '-d', '--download',
        action='store_true',
        help="Download dataset from kaggle."
    )

    # -f, --force: overwrite
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help="Do action forcefully. (Default: False)"
    )

    # -lp, --load_path: create path cache csv
    parser.add_argument(
        '-lp', '--load_path',
        action='store_true',
        help="Create and load path cache csv file."
    )

    return parser.parse_args()

def main():
    """
    main
    """

    args = parse_arguments()

    if args.download:
        print ("\nStarting Data download")

        if args.force:
            print("\nOverwrite exsisting data...")

        # dataset_path = download_dataset(force=args.force)
        dataset_path = DOWNLOAD_DIR # above not working, manually download to the data folder

        if dataset_path is None:
            print("\nError occured, system terminating...")
            sys.exit(1)

        print(f"Dataset is ready at '{dataset_path}'.")

    if args.load_path:
        print ("\nStarting path creation")

        csv_paths = save_raw_h5_file_paths(
            force_rebuild=args.force
        )

        if not csv_paths:
            print("\nError occured, system terminating...")
            sys.exit(1)

        print(f"\nDataset load ready.")
        print(f"Path file available at '{csv_paths}'")

if __name__ == "__main__":
    main()