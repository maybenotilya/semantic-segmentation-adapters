import cv2
import logging

from pathlib import Path
from typing import Dict, Optional, List

from common.adapter.adapter_base import AdapterBase
from common.utils.metrics import AVAILABLE_METRICS


class Inferencer:
    def __init__(
        self,
        adapter: AdapterBase,
        classes: List[str],
        images_dir: Path,
        output_dir: Path,
        masks_dir: Path = None,
    ):
        self._adapter = adapter
        self._classes = classes
        self._images_dir = images_dir
        self._output_dir = output_dir
        self._masks_dir = masks_dir

        self._metrics = None

    def _infer(self):
        if self._masks_dir is not None:
            self._metrics = {
                class_name: {metric_name: 0 for metric_name in AVAILABLE_METRICS}
                for class_name in self._classes
            }
            images_count = sum(1 for _ in self._images_dir.iterdir())

        for image_path in self._images_dir.iterdir():
            logging.info(f" Image {image_path}")
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            prediction = self._adapter.process(image)

            if self._masks_dir is not None:
                logging.info(" Evaluating metrics...")
                mask_path = self._masks_dir / Path(image_path.stem).with_suffix(
                    image_path.suffix
                )
                mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
                for class_label, class_name in enumerate(self._classes, 1):
                    for metric_name, metric_fn in AVAILABLE_METRICS.items():
                        self._metrics[class_name][metric_name] += (
                            metric_fn(prediction, mask, class_label) / images_count
                        )

            output_path = self._output_dir / Path(image_path.stem).with_suffix(".png")
            logging.info(f" Saving to {output_path}")
            cv2.imwrite(output_path, prediction)

    def __call__(self, *args, **kwargs):
        return self._infer(*args, **kwargs)

    @property
    def metrics(self) -> Optional[Dict[str, Dict[str, float]]]:
        return self._metrics
