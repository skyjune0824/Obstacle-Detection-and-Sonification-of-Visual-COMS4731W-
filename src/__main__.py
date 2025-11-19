import argparse

# MDE Pipeline
from src.MDE.pipeline import MDE_Pipeline

def structure_from_motion_pipeline(source):
    """
    The following function marks the pipeline for creating a minimal point cloud
    from multiple images in motion, performing segmentation, and synthesizing spatial audio.

    Source: Path to video we desire to perform our pipeline on.
    Debug: Toggles Verbose

    """

    print(f"Performing SFM on video located at: {source}")

    return NotImplemented

def main():
    # Initialize Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("input_video", help="Path to the input source video.")

    parser.add_argument(
        "--mde",
        action="store_true",
        help="Monocular Depth Estimation -> Spatial Audio",
    )

    parser.add_argument(
        "--sfm",
        action="store_true",
        help="Structure From Motion -> Spatial Audio",
    )

    args = parser.parse_args()

    # Test
    if args.mde:
        # Initialize Pipeline
        MDE = MDE_Pipeline(rate = 4)

        # Call Pipeline
        MDE.pipeline(args.input_video)
    elif args.sfm:
        structure_from_motion_pipeline(args.input_video)

if __name__ == "__main__":
    main()