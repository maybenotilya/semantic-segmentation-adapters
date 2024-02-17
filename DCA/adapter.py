# Copyright (c) 2022, Ilya Syresenkov, Kirill Ivanov and Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import ever
import torch
import cv2

from pathlib import Path
from albumentations import Normalize, Compose

from common.adapter.adapter_base import AdapterBase
from DCA.utils.my_tools import pre_slide
from DCA.module.Encoder import Deeplabv2


class DcaAdapter(AdapterBase):
    """
    Adapter to work with DCA algorithm
    """

    def __init__(
            self,
            factor: int,
            model: Path,
            device: str
    ):
        """
        Args:
            factor (int): factor to divide image into patches
            model (Path): path to pretrained model weights
            device (str): PyTorch device to run network on
        """
        self._factor = factor
        self._model = model
        self._device = device

    def process(self, image: np.ndarray):
        shape = image.shape[:-1]
        image = self._transform_image(image)
        model = self._build_model()
        raw_predictions = self._process(model, image)
        predictions = self._postprocess_predictions(raw_predictions)
        return predictions

    def _transform_image(self, image: np.ndarray):
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        transformer = Compose([
            Normalize(mean=mean,
                      std=std,
                      max_pixel_value=1, always_apply=True),
            ever.preprocess.albu.ToTensor()
        ], is_check_shapes=False)
        blob = transformer(image=image)
        image = blob["image"].to(self._device)
        image = image[None, :]
        return image

    def _build_model(self):
        model = Deeplabv2(dict(
            backbone=dict(
                resnet_type="resnet50",
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=True,
            cascade=False,
            use_ppm=True,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                fc_dim=2048,
            ),
            inchannels=2048,
            num_classes=7
        )).to(self._device)
        model_state_dict = torch.load(self._model,
                                      map_location=self._device)
        model.load_state_dict(model_state_dict, strict=True)
        model.eval()
        return model

    def _process(self, model, image):
        shape = image.shape[2], image.shape[3]
        with torch.no_grad():
            cls = pre_slide(
                model=model,
                image=image,
                num_classes=7,
                tile_size=(image.shape[2] // self._factor, image.shape[3] // self._factor),
                tta=True,
                device=self._device
            )
            cls = cls.argmax(dim=1).to(self._device).numpy().reshape(shape).astype(np.uint8)
            return cls

    def _postprocess_predictions(self, raw_predictions):
        return raw_predictions
