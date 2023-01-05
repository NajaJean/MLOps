# Base image
FROM python:3.7-slim

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY src/ src/
COPY data/ data/

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/train_model.py"]
