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
import logging

from pathlib import Path
from skimage.io import imread

from utils import get_args
from adapter import DcaAdapter

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    args = get_args()
    if args.device.lower() == "cpu":
        device = "cpu"
    elif args.device.lower() in ("cuda", "gpu"):
        device = "cuda"
    else:
        device = "cuda"
        logging.error(f" Failed to recognize device {args.device}, default GPU is set")

    adapter = DcaAdapter(factor=args.factor, model=args.model, device=device)

    input_dir = args.input
    output_dir = args.output

    for file_path in input_dir.iterdir():
        logging.info(f" Image {file_path}")
        image = imread(file_path)
        logging.info(f"Image shape: {image.shape}")
        mask = adapter.process(image)
        output_path = output_dir / Path(file_path.stem).with_suffix('.npy')
        logging.info(f" Saving to {output_path}")
        np.save(output_path, mask)