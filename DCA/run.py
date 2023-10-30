from cv2 import imwrite
from skimage.io import imread
import numpy as np

from pathlib import Path
import os
import argparse

from adapter_utils import max_power_of_2, get_args
from adapter import DcaAdapter


if __name__ == "__main__":
    print("Starting work")
    args = get_args()

    if args.device.lower() == 'cuda' or args.device.lower() == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    adapter = DcaAdapter(factor=args.factor, mode=args.mode.lower())

    input_path = Path(__file__).parent / "input"
    output_path = Path(__file__).parent / "output"

    for fname in os.listdir(input_path):
        print(fname)
        im = imread(input_path / fname)
        p = max_power_of_2(min(im.shape[0], im.shape[1]))
        adapter.image_size = p
        mask = adapter.process(im)
        np.save(output_path / fname.split('.')[0], mask)
