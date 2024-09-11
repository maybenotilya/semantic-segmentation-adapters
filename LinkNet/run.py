import cv2
import logging

from pathlib import Path

from common.utils.common_utils import get_device_by_name
from common.utils.infer import Inferencer
from adapter import LinkNetAdapter
from utils import get_args


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    args = get_args()
    device = get_device_by_name(args.device)

    adapter = LinkNetAdapter(model_path=args.model, device=device)
    classes = ["road"]

    images_dir = args.images
    output_dir = args.output
    masks_dir = args.masks if args.metrics else None

    infer = Inferencer(adapter, classes, images_dir, output_dir, masks_dir)
    infer()

    metrics = infer.metrics
    if metrics is not None:
        for class_name, class_metrics in metrics.items():
            logging.info(f"{class_name}\n{class_metrics}    ")
