FROM python:3.10

ARG ALGORITHM_FOLDER=/DCA

WORKDIR $ALGORITHM_FOLDER

COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

ARG CKPT_PATH=/DCA/weights

RUN pip install gdown \
    && mkdir $CKPT_PATH \
    && gdown "https://drive.google.com/uc?export=download&id=1ZPsIJdEFLDvVSvguOyJj1WiCTANZiYs0" -O $CKPT_PATH/Rural.pth \
    && gdown "https://drive.google.com/uc?export=download&id=1R6j6dzIf-Y5S48_EHEMtKvUYawmyt6Ul" -O $CKPT_PATH/Urban.pth

COPY . .

ENTRYPOINT [ "python", "run.py" ]
