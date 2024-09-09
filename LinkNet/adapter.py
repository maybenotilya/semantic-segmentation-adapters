import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp
import json

from pathlib import Path

from common.adapter.adapter_base import AdapterBase


class LinkNetAdapter(AdapterBase):
    """
    Adapter to work with LinkNet algorithm
    """

    def __init__(self, model_path: Path, device: str):
        """
        Args:
            device (str): PyTorch device to run network on
        """
        self._model_path = model_path
        self._model = None
        self._device = device
        with open(model_path / "config.json") as config_file:
            model_cfg = json.load(config_file)
        self._encoder_name = model_cfg["encoder_name"]
        self._encoder_weights = model_cfg["encoder_weights"]

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            self._encoder_name, pretrained=self._encoder_weights
        )

        self._image_shape = image.shape[:2]
        image = cv2.resize(image, (512, 512))

        image = preprocessing_fn(image)
        image = image.transpose(2, 0, 1).astype("float32")
        return torch.from_numpy(image[np.newaxis, ...])

    def _build_model(self):
        if self._model is None:
            self._model = smp.from_pretrained(self._model_path)
        return self._model

    def _process(self, model, image: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            pred = model(image)
        return pred.detach().cpu().numpy()

    def _postprocess_predictions(self, raw_predictions: np.ndarray) -> np.ndarray:
        pred = (raw_predictions > 0.5).squeeze().astype("float32")
        pred = cv2.resize(pred, self._image_shape[::-1])
        pred = (pred > 0.5).astype(int)
        pred[pred == 1] = 240
        return pred
