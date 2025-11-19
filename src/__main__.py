import argparse

# MDE Pipeline
from src.MDE.pipeline import monocular_depth_estimation_pipeline

def structure_from_motion_pipeline(source, debug: bool):
    """
    The following function marks the pipeline for creating a minimal point cloud
    from multiple images in motion, performing segmentation, and synthesizing spatial audio.

    Source: Path to video we desire to perform our pipeline on.
    Debug: Toggles Verbose

    """

    if debug: 
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show Verbose",
    )

    args = parser.parse_args()

    # Test
    if args.mde:
        monocular_depth_estimation_pipeline(args.input_video, args.debug)
    elif args.sfm:
        structure_from_motion_pipeline(args.input_video, args.debug)

if __name__ == "__main__":
    main()