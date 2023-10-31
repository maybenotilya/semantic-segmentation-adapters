import torch

from argparse import ArgumentParser
from pathlib import Path


def max_power_of_2(num : int):
    p: int = 2
    while p * 2 <= num:
        p *= 2
    return p


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="Urban", help="Mode to segment images with: Urban or Rural, default is Urban")
    parser.add_argument("-d", "--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Which device to run network on, default is GPU if available, otherwise CPU")
    parser.add_argument("-f", "--factor", type=int, default=2, help="Factor shows how images must be scaled to create patches, for factor = n there will be n^2 patches, default is 2")
    return parser.parse_args()

