# Stage 1 
FROM python:3.7-slim as builder

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1-mesa-glx && \
    apt update && \
    apt install --no-install-recommends -y libgtk2.0-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --user -r requirements.txt

#Stage 2
FROM debian:buster-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local/lib/python3.7/site-packages /usr/local/lib/python3.7/dist-packages
COPY Data/Model_Weights ./Data/Model_Weights
COPY Utils/utils.py .
COPY 2_Training/src .
COPY 4_Deploy/ .
CMD [ "export" , "FLASK_APP=webService.py" ]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]