import sys
import os
import argparse
from preprocess.data_management import download_dataset, DOWNLOAD_DIR

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
        help="Forcefully redownload dataset from kaggle. (Default: False)"
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




if __name__ == "__main__":
    main()