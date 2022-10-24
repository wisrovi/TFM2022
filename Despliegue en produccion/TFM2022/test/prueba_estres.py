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



requests.adapters.DEFAULT_RETRIES = 5 # increase retries number




# initialize the Keras REST API endpoint URL along with the input
# image path

# leer la carpeta de ejecucion


KERAS_REST_API_URL = "http://localhost:52001/RNA?test=YES"
BASE_FOLDER = os.chdir(os.path.dirname(os.path.abspath(__file__)))
if BASE_FOLDER is None:
    BASE_FOLDER = os.getcwd() + os.sep 
IMAGE_PATH = BASE_FOLDER +  "demo" + os.sep + "dat_85.wav"

print("Data: ", IMAGE_PATH)
print("Keras REST API URL: ", KERAS_REST_API_URL)

headers={
'Referer': 'https://itunes.apple.com',
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
}




# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 1
SLEEP_COUNT = 0.05

conteo_ok = 0
conteo_bad = 0


from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
def call_predict_endpoint(n):
    global conteo_ok, conteo_bad

    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"file1": image}

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    r = session.post(KERAS_REST_API_URL, files=payload)
    print(a.status_code)

    print("*" * 10)


    # submit the request
    #r = requests.post(KERAS_REST_API_URL, files=payload, verify=False, headers=headers, timeout=(2, 5))

    if r.status_code == 200:
        data = None
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
        print("[INFO] thread {} FAILED".format(n), r.status_code)
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

