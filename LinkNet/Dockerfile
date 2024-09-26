FROM python:3.12

ARG ALGORITHM_FOLDER=/LinkNet

WORKDIR ${ALGORITHM_FOLDER}

COPY ${ALGORITHM_FOLDER}/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

ARG MODEL_PATH=${ALGORITHM_FOLDER}/models

RUN python -m pip install --no-cache-dir gdown \
    && mkdir ${MODEL_PATH} \
    && gdown --folder   --fuzzy "https://drive.google.com/drive/folders/1qYG5UPk5ORVSs3hJldAwwkOxRXep8amJ?usp=drive_link" -O ${MODEL_PATH}/roads-seg-model

COPY ${ALGORITHM_FOLDER} .

COPY common common

ENTRYPOINT [ "python", "run.py" ]
