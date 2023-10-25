from pathlib import Path
from adapter import DcaAdapter, max_power_of_2
from cv2 import imwrite
from skimage.io import imread
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=False, help="Network mode: Urban or Rural.")
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting work")
    Path("./output").mkdir(exist_ok=True, parents=True)
    args = get_args()
    mode = "Urban"
    adapter = DcaAdapter(mode=mode)
    for fname in os.listdir("./input"):
        print(fname)
        im = imread(str("./input") + f"/{fname}")
        p = max_power_of_2(min(im.shape[0], im.shape[1]))
        adapter.set_size(p)
        mask = adapter.process(im)
        imwrite(str("./output") + f"/{fname.replace('jpg', 'png').replace('tif', 'png').replace('jpeg', 'png')}", mask)
