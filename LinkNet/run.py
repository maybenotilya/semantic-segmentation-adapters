import cv2
import logging

from pathlib import Path

from common.utils.common_utils import get_device_by_name
from adapter import LinkNetAdapter
from utils import get_args


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    args = get_args()
    device = get_device_by_name(args.device)

    adapter = LinkNetAdapter(model_path=args.model, device=device)

    images_dir = args.images
    masks_dir = args.masks
    output_dir = args.output

    for image_path in images_dir.iterdir():
        logging.info(f" Image {image_path}")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        mask = adapter.process(image)

        output_path = output_dir / Path(image_path.stem).with_suffix(".png")
        logging.info(f" Saving to {output_path}")
        cv2.imwrite(output_path, mask)
