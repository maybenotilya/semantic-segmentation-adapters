FROM python:3.10

COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /DCA

COPY weights weights

COPY utils utils 

COPY module module 

COPY data data

COPY adapter_base.py adapter_base.py

COPY adapter.py adapter.py

COPY run.py run.py

RUN mkdir input

RUN mkdir output

ENTRYPOINT [ "python", "run.py" ]
