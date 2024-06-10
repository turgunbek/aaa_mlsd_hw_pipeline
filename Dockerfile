FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade "pip==22.3"

WORKDIR /app

COPY ./requirements.txt $WORKDIR/

RUN pip3 install --no-cache-dir -r $WORKDIR/requirements.txt

COPY ./main.py $WORKDIR/
COPY ./models_and_utils.py $WORKDIR/

# Копирование папки data
COPY ./data $WORKDIR/data

RUN PYTHONPATH="$WORKDIR:$PYTHONPATH"
