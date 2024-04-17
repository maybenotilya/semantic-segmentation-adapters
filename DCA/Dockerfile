FROM python:3.10

ARG ALGORITHM_FOLDER=/DCA

WORKDIR $ALGORITHM_FOLDER

COPY $ALGORITHM_FOLDER/requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

ARG CKPT_PATH=/DCA/weights

RUN pip install gdown \
    && mkdir $CKPT_PATH \
    && gdown "https://drive.google.com/uc?export=download&id=1ZPsIJdEFLDvVSvguOyJj1WiCTANZiYs0" -O $CKPT_PATH/Rural.pth \
    && gdown "https://drive.google.com/uc?export=download&id=1R6j6dzIf-Y5S48_EHEMtKvUYawmyt6Ul" -O $CKPT_PATH/Urban.pth \
    && gdown --fuzzy "https://drive.google.com/file/d/14Krod0sXaBJdxoCex-mC85Afb_nzS7Ig/view?usp=drive_link" -O $CKPT_PATH/Urban1101.pth \
    && gdown --fuzzy "https://drive.google.com/file/d/1KLoj_sorw_qkXQeQnWaaIE99vhspoQwA/view?usp=drive_link" -O $CKPT_PATH/Urban2101.pth

COPY  $ALGORITHM_FOLDER .

COPY ./common ./common

ENTRYPOINT [ "python", "run.py" ]
