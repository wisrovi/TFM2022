#FROM python:3.8-slim-buster
FROM tiangolo/uvicorn-gunicorn:python3.8-slim

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

LABEL maintainer="William Rodriguez <wisrovi.rodriguez@gmail.com>"

RUN apt-get update
RUN apt-get install libsndfile1-dev -y

RUN pip3 install pickle-mixin
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install librosa
RUN pip3 install redis
RUN pip3 install fastapi
RUN pip3 install uvicorn
RUN pip3 install gunicorn
RUN pip3 install python-multipart
RUN pip3 install jinja2
#RUN pip3 install ProcessAudio


COPY . .
