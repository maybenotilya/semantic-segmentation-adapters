# DCA
This folder contains adapter and Dockerfile for [DCA](https://github.com/Luffy03/DCA) algorithm

## Building Docker Image
Build docker image from parent directory using `Dockerfile`:
```
docker build . -t dca -f DCA/Dockerfile
```

## Running Docker Container
To run the container use the following command:
```
docker run --rm \
-v <IMAGES_PATH>:/DCA/input \
-v <OUTPUT_PATH>:/DCA/output \
dca [OPTIONAL_ARGS]
```

Optional arguments are:
```
-f, --factor       Factor shows how images must be scaled to create patches, for factor = n there will be n^2 patches (default: 2)
-m, --model        Pretrained model path (default: weights/Urban.pth)
-d, --device       Which device to run network on (default: GPU)
-i, --input        Images input directory
-o, --output       Masks output directory
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine and `<OUTPUT_PATH>` is the path where the result numpy arrays will be saved. 
