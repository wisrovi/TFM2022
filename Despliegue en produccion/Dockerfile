FROM python:3.8-slim-buster

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

CMD ["python3", "/app/api_rest.py"]
