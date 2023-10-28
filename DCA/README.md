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
dca
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine and `<OUTPUT_PATH>` is the path where the results will be saved. 

Also you can use `colors.py` script to color all masks in folder (will be added to pipeline later):
```
python3 colors.py --input <INPUT_PATH> --output <OUTPUT_PATH>
```
