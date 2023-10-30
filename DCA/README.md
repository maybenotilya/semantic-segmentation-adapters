# DCA
This folder contains adapter and Dockerfile for [DCA](https://github.com/Luffy03/DCA) algorithm

## Building Docker Image
Build docker image using `Dockerfile`:
```
docker build . -t dca -f Dockerfile
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
-m, --mode       Network mode to segment model, Urban or Rural, default is Urban
-d, --device     Which device to run network on, default is GPU if GPU is available, otherwise CPU
-f, --factor     Factor shows how images must be scaled to create patches, for factor = n there will be n^2 patches, default is 2
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine and `<OUTPUT_PATH>` is the path where the result numpy arrays will be saved. 
