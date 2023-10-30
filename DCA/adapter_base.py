import numpy as np

from abc import ABC, abstractmethod

class AdapterBase(ABC):
    """Adapter accepts image as numpy array and returns mask as image in numpy array"""

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
