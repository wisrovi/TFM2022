FROM python:3.8-slim-buster

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

LABEL maintainer="William Rodriguez <wisrovi.rodriguez@gmail.com>"

WORKDIR /app

RUN apt-get update
RUN apt-get install libsndfile1-dev -y

RUN pip3 install pickle-mixin
RUN pip3 install pandas
RUN pip3 install redis
RUN pip3 install matplotlib
RUN pip3 install librosa


COPY . .


CMD ["python3", "/app/service_model.py"]