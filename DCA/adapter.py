import numpy as np

from PIL import Image
import argparse
from pathlib import Path

from adapter_base import AdapterBase
from utils.tools import *
from utils.my_tools import *
from module.Encoder import Deeplabv2
from albumentations import Normalize
from albumentations import *
import ever as er

def max_power_of_2(num : int):
    p: int = 2
    while p * 2 <= num:
        p *= 2
    return p


class DcaAdapter(AdapterBase):
    def __init__(self, size=1024, factor=2, mode="Urban"):
        self.size = size
        self.factor = factor
        self.mode = mode

    def set_size(self, size):
        self.size = size

    def process(self, image: np.ndarray):
        # image - RGB
        image = self._transform_image(image)
        model = self._build_model()
        raw_predictions = self._process(model, image)
        predictions = self._postprocess_predictions(raw_predictions)
        return predictions

    # requires MxMx3 image
    def _transform_image(self, image: np.ndarray):
        image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize((self.size, self.size))
        image = np.array(image)
        print(image.shape)
        mask = np.zeros([512, 512])
        transformer = Compose([
            Normalize(mean=(73.53223948, 80.01710095, 74.59297778),
                    std=(41.5113661, 35.66528876, 33.75830885),
                    max_pixel_value=1, always_apply=True),
            # Normalize(mean=(123.675, 116.28, 103.53),
            #           std=(58.395, 57.12, 57.375),
            #           max_pixel_value=1, always_apply=True),
            er.preprocess.albu.ToTensor()
        ], is_check_shapes=False)
        blob = transformer(image=image, mask=mask)
        image = blob['image'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        image = image[None, : ]
        return image

    def _build_model(self):
        model = Deeplabv2(dict(
            backbone=dict(
                resnet_type='resnet50',
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
        )).to('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_path = {"Urban" : "./weights/URBAN_0.4635.pth", 
                     "Rural" : "./weights/RURAL_0.4517.pth"}
        model_state_dict = torch.load(ckpt_path[self.mode], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(model_state_dict, strict=True)
        model.eval()
        return model


    def _process(self, model, image):
        with torch.no_grad():
            cls = pre_slide(model, image, num_classes=7, tile_size=(self.size // self.factor, self.size // self.factor), tta=True)
            cls = cls.argmax(dim=1).to('cuda' if torch.cuda.is_available() else 'cpu').numpy()
            return cls
    
    def _postprocess_predictions(self, raw_predictions):
        return raw_predictions.reshape(self.size, self.size).astype(np.uint8)
