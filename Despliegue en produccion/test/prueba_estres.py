# USAGE
# python stress_test.py

# import the necessary packages
import os
from threading import Thread
import requests
import time


import logging
from time import strftime
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler("../log/stress_test.log"),
                              logging.StreamHandler()])


# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:52000/RNA"
IMAGE_PATH = "demo/dat_85.wav"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 1_000
SLEEP_COUNT = 0.04

conteo_ok = 0
conteo_bad = 0

def call_predict_endpoint(n):
    global conteo_ok, conteo_bad

    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"file1": image}

    # submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload, data={"test":"YES"})

    if r.status_code == 200:
        try:
            data = r.json()[0].get("instruments_predict")
            #list to string
            data = " - ".join(data)
            print(data)
        except:
            print("Error: ", r.text)

        logging.info(f" * Test {str(n)}: [{IMAGE_PATH.split(os.sep)[-1]}]: {data}") 

        print("[INFO] thread {} OK".format(n))
        conteo_ok += 1
    else:
        print("[INFO] thread {} FAILED".format(n))
        logging.error("falló la peticion")
        conteo_bad += 1



t_inicial = time.time()
# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

time.sleep(10)


t_final = time.time()
duracion_segundos = t_final - t_inicial
print('Tiempo transcurrido (en segundos): {}'.format(duracion_segundos))
print("Ok:", conteo_ok, " - bad:", conteo_bad)
logging.info(f"Total test: {str(NUM_REQUESTS)} - tiempo: {str(round(duracion_segundos, 3))} -  Resultado: OK: {str(conteo_ok)} / BAD: {str(conteo_bad)}")

# insert a long sleep so we can wait until the server is finished
# processing the images

