from cv2 import imwrite
from skimage.io import imread

from pathlib import Path
import os
import argparse

from adapter_utils import max_power_of_2, get_args, colour_mask
from adapter import DcaAdapter


if __name__ == "__main__":
    print("Starting work")
    args = get_args()
    adapter = DcaAdapter(mode=args.mode)
    for fname in os.listdir("/DCA/input"):
        print(fname)
        im = imread(str("./input") + f"/{fname}")
        p = max_power_of_2(min(im.shape[0], im.shape[1]))
        adapter.image_size = p
        mask = adapter.process(im)
        if args.colour:
            mask = colour_mask(mask)
        imwrite(str("./output") + f"/{fname.replace('jpg', 'png').replace('tif', 'png').replace('jpeg', 'png')}", mask)
