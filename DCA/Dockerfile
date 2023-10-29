FROM python:3.10

ARG ALGORITHM_FOLDER=DCA

WORKDIR /DCA

COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install gdown \
    && mkdir weights \
    && gdown "https://drive.google.com/uc?export=download&id=1ZPsIJdEFLDvVSvguOyJj1WiCTANZiYs0" -O ./weights/RURAL_0.4517.pth \
    && gdown "https://drive.google.com/uc?export=download&id=1R6j6dzIf-Y5S48_EHEMtKvUYawmyt6Ul" -O ./weights/URBAN_0.4635.pth

COPY . .

ENTRYPOINT [ "python", "run.py" ]
