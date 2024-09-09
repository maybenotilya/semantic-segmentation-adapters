import torch
import logging


def get_device_by_name(device_name: str):
    if device_name.lower() == "cpu":
        return torch.device("cpu")
    elif device_name.lower() in ("cuda", "gpu"):
        return torch.device("cuda")
    else:
        logging.warning(f"Failed to recognize device {device_name}, default GPU is set")
        return torch.device("cuda")
