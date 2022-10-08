FROM python:3.8-slim-buster
WORKDIR /app

RUN pip3 install flask
RUN pip3 install ProcessAudio
RUN pip3 install pickle-mixin
RUN pip3 install pandas

COPY src .

RUN apt-get update
RUN apt-get install libsndfile1-dev -y

CMD ["python3", "./service.py"]

