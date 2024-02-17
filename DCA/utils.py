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

from argparse import ArgumentParser
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--factor", type=int, default=2,
                        help="Factor shows how images must be scaled to create patches, for factor = n there will be "
                             "n^2 patches (default: 2)")
    parser.add_argument("-m", "--model", type=Path, default=Path("/", "DCA", "weights", "Urban.pth"),
                        help="Pretrained model path (default: weights/Urban.pth)")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Which device to run network on (default: GPU)")
    parser.add_argument("-i", "--input", type=Path, default=Path("/", "DCA", "input"),
                        help="Images input folder")
    parser.add_argument("-o", "--output", type=Path, default=Path("/", "DCA", "output"),
                        help="Masks output folder")
    return parser.parse_args()
