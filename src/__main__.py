import argparse

def monocular_depth_estimation_pipeline(source):
    """
    The following function marks the pipeline for estimating a depth map,
    performing segmentation, and synthesizing spatial audio. 
    """

    # DEBUG
    print(f"MDE Path: {source}")

    return NotImplemented

def structure_from_motion_pipeline(source):
    """
    The following function marks the pipeline for creating a minimal point cloud
    from multiple images in motion, performing segmentation, and synthesizing spatial audio. 
    """

    # DEBUG
    print(f"SFM Path: {source}")

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
        monocular_depth_estimation_pipeline(args.input_video)
    elif args.sfm:
        structure_from_motion_pipeline(args.input_video)

if __name__ == "__main__":
    main()