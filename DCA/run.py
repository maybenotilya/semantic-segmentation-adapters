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
import os

from pathlib import Path
from skimage.io import imread

from adapter_utils import get_args
from adapter import DcaAdapter


def main():
    print("Starting work")

    args = get_args()
    if args.device.lower() == "cpu":
        device = "cpu"
    elif args.device.lower() in ("cuda", "gpu"):
        device = "cuda"
    else:
        print(f"Failed to recognize device {args.device}")
        return

    adapter = DcaAdapter(factor=args.factor, model=args.model, device=device)

    input_path = Path(__file__).parent / "input"
    output_path = Path(__file__).parent / "output"

    for fname in os.listdir(input_path):
        print(fname)
        image = imread(input_path / fname)
        mask = adapter.process(image)
        output_name = '.'.join(fname.split('.')[:-1]) if fname.count('.') != 0 else fname
        np.save(output_path / output_name, mask)


if __name__ == "__main__":
    main()
