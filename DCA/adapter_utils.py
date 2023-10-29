import numpy as np

import argparse

def max_power_of_2(num : int):
    p: int = 2
    while p * 2 <= num:
        p *= 2
    return p


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="Urban", help="Mode to segment images with: Urban or Rural, default is Urban")
    parser.add_argument("-c", "--colour", type=bool, default=False, help="Colours result images, default is False")
    return parser.parse_args()

def colour_mask(image: np.ndarray):
    colors = {
        0 : (255, 255, 255), # background
        1 : (255, 0, 0), # building
        2 : (255, 255, 0), # road
        3 : (0, 0, 255), # water
        4 : (128, 0, 128), # barren
        5 : (0, 128, 0), # forest
        6 : (141, 85, 36), # agriculture
    }
    new_image = np.ndarray(shape=(image.shape[0], image.shape[1], 3))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_image[x, y] = colors[image[x, y]]
    return new_image


