import numpy as np
import torch
import logging

from typing import List


def get_device_by_name(device_name: str):
    if device_name.lower() == "cpu":
        return torch.device("cpu")
    elif device_name.lower() in ("cuda", "gpu"):
        return torch.device("cuda")
    else:
        logging.warning(f"Failed to recognize device {device_name}, default GPU is set")
        return torch.device("cuda")


def get_class_from_mask(*masks: np.ndarray, class_label: int) -> List[np.ndarray]:
    return [(mask == class_label).astype(int) for mask in masks]
