FROM tensorflow/tensorflow
WORKDIR /app

RUN pip3 install flask
RUN pip3 install ProcessAudio
RUN pip3 install pickle

COPY src .

CMD ["python3", "./service.py"]

