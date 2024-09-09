from argparse import ArgumentParser
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=False,
        default=Path("/LinkNet/models/roads-seg-model"),
        help="Path to pretrained model (in segmentation-models format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run model on. Default is Cuda (GPU)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=False,
        default=Path("/LinkNet/images"),
        help="Images input folder",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        required=False,
        default=Path("/LinkNet/masks"),
        help="Masks input folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        default=Path("/LinkNet/output"),
        help="Masks output folder",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        required=False,
        default=True,
        help="Evaluate metrics on provided data. Masks path must be provided.",
    )

    return parser.parse_args()
