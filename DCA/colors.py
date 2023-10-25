from PIL import Image
import os
import argparse
from pathlib import  Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input images path.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output images path.")
    return parser.parse_args()

def color_image():
    colors = {
        (0, 0, 0) : (255, 255, 255), # background
        (1, 1, 1) : (255, 0, 0), # building
        (2, 2, 2) : (255, 255, 0), # road
        (3, 3, 3) : (0, 0, 255), # water
        (4, 4, 4) : (128, 0, 128), # barren
        (5, 5, 5) : (0, 128, 0), # forest
        (6, 6, 6) : (141, 85, 36), # agriculture
    }

    args = get_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for fname in os.listdir(args.input):
        img = Image.open(str(args.input) + f"/{fname}").convert("RGB")
        pixels = img.load()
        x, y = img.size
        for i in range(x):
            for j in range(y):
                pixels[i, j] = colors[pixels[i, j]]
                if pixels[i, j] == (0, 0, 0):
                    print(i, j)
        img.save(str(args.output) + f"/{fname}")



if __name__ == '__main__':
    color_image()
