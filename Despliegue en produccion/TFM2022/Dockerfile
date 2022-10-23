FROM python:3.8-slim-buster

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip3 install flask
#RUN pip3 install ProcessAudio
RUN pip3 install pickle-mixin
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install librosa
RUN pip3 install redis

RUN apt-get update
RUN apt-get install libsndfile1-dev -y

COPY src .

