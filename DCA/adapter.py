import numpy as np
from albumentations import Normalize, Compose
import ever
import torch

from PIL import Image
import argparse
from pathlib import Path

from adapter_base import AdapterBase
from DCA.utils.my_tools import pre_slide
from DCA.module.Encoder import Deeplabv2


class DcaAdapter(AdapterBase):
    
    def __init__(self, image_size=1024, factor=2, mode="Urban"):
        '''
        Image Size -- size to scale image to (algorithm requires square images)
        Factor -- algorithm runs segmentation on square patches of image, factor shows how to scale each side of image to make these patches (if factor=n, there will be n^2 patches)
        '''

        self._image_size = image_size
        self._factor = factor
        self._mode = mode

    @property
    def image_size(self):
        return self._image_size
    
    @image_size.setter
    def image_size(self, image_size):
        self._image_size = image_size

    @property
    def factor(self):
        return self._factor
    
    @factor.setter
    def factor(self, factor):
        self._factor = factor

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        self._mode = mode

    def process(self, image: np.ndarray):
        # image - RGB
        image = self._transform_image(image)
        model = self._build_model()
        raw_predictions = self._process(model, image)
        predictions = self._postprocess_predictions(raw_predictions)
        return predictions

    '''Input: MxMx3 image'''
    def _transform_image(self, image: np.ndarray):
        image = Image.fromarray(image)
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)
        print(image.shape)
        transformer = Compose([
            Normalize(mean=(73.53223948, 80.01710095, 74.59297778),
                    std=(41.5113661, 35.66528876, 33.75830885),
                    max_pixel_value=1, always_apply=True),
            # Normalize(mean=(123.675, 116.28, 103.53),
            #           std=(58.395, 57.12, 57.375),
            #           max_pixel_value=1, always_apply=True),
            ever.preprocess.albu.ToTensor()
        ], is_check_shapes=False)
        blob = transformer(image=image)
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
        ckpt_path = {"Urban" : "/DCA/weights/URBAN_0.4635.pth", 
                     "Rural" : "/DCA/weights/RURAL_0.4517.pth"}
        model_state_dict = torch.load(ckpt_path[self.mode], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(model_state_dict, strict=True)
        model.eval()
        return model


    def _process(self, model, image):
        with torch.no_grad():
            cls = pre_slide(model, image, num_classes=7, tile_size=(self.image_size // self.factor, self.image_size // self.factor), tta=True)
            cls = cls.argmax(dim=1).to('cuda' if torch.cuda.is_available() else 'cpu').numpy()
            return cls
    
    def _postprocess_predictions(self, raw_predictions):
        return raw_predictions.reshape(self.image_size, self.image_size).astype(np.uint8)
