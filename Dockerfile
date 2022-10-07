FROM tensorflow/tensorflow
WORKDIR /app

RUN pip3 install flask
RUN pip3 install keras
RUN pip3 install Pillow

COPY src .

CMD ["python3", "./service.py"]

