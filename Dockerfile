# Stage 1 
FROM python:3.7-slim as builder
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt update && \
    apt install --no-install-recommends -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --user -r requirements.txt

#Stage 2
FROM debian:buster-slim
ARG IS_TINY=no
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip python3-setuptools libgtk2.0-dev libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --user flask==1.1.2

COPY --from=builder /root/.local/lib/python3.7/site-packages /usr/local/lib/python3.7/dist-packages
COPY Data/Model_Weights/trained_weights_final.h5 ./Data/Model_Weights/
COPY Utils/utils.py .
COPY 2_Training/src .
COPY 4_Deploy/ .
ENV IS_TINY_MODEL=$IS_TINY
ENV FLASK_APP=webService.py
EXPOSE 5000
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]