import argparse

# MDE Pipeline
from src.MDE.pipeline import MDE_Pipeline
from src.SFM.pipeline import SFM_Pipeline

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
        "--thresh",
        type=int,
        default=5,
        help="Structure From Motion -> Spatial Audio",
    )

    args = parser.parse_args()

    # Test
    if args.mde:
        # Initialize Pipeline
        MDE = MDE_Pipeline(rate=60, threshold=args.thresh)

        # Call Pipeline
        MDE.pipeline(args.input_video)
    elif args.sfm:
        # Initialize Pipeline
        SFM = SFM_Pipeline()

        # Call Pipeline
        SFM.pipeline(args.input_video)

if __name__ == "__main__":
    main()