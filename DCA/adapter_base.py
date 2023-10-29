import numpy as np

import argparse
from pathlib import Path
from abc import ABC, abstractmethod

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input images path.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path where to save resulting masks.")
    return parser.parse_args()


class AdapterBase(ABC):
    """Adapter accepts image as numpy array and returns mask as image in numpy arrayr"""

    def process(self, image):
        # image - RGB
        image = self._transform_image(image)
        model = self._build_model()
        raw_predictions = self._process(model, image)
        predictions = self._postprocess_predictions(raw_predictions)
        return predictions

    @abstractmethod
    def _transform_image(self, image: np.ndarray):
        pass

    @abstractmethod
    def _process(self, model, image):
        pass


    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _postprocess_predictions(self, raw_predictions):
        pass
