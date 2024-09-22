# LinkNet
This folder contains adapter and Dockerfile for LinkNet algorithm for roads semantic segmentation

## Building Docker Image
Build docker image from parent directory using `Dockerfile`:
```
docker build . -t linknet -f LinkNet/Dockerfile
```

## Running Docker Container
To run the container use the following command:
```
docker run --rm \
-v <IMAGES_PATH>:/LinkNet/images \
-v <OUTPUT_PATH>:/LinkNet/output \
[-v <MASKS_PATH>:/LinkNet/masks]
linknet [OPTIONAL_ARGS]
```

Optional arguments are:
```
--model        Pretrained model path (default: /LinkNet/models/roads-seg-model)
--device       Which device to run network on (default: GPU)
--images       Images input directory (default: /LinkNet/images)
--output       Output directory (default: /LinkNet/output)
--masks        Ground Truth masks directory (default: /LinkNet/masks)
--metrics      Use this flag to evaluate metrics. Masks directory must be provided (default: False)
```

Here `<IMAGES_PATH>` is the path where your image dataset is stored on the host machine, `<OUTPUT_PATH>` is the path where the result masks will be saved and `<MASKS_PATH>` is the path where ground truth masks are stored.
