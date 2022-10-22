import json
import numpy as np
import config.settings as settings
from functools import wraps
import time
import redis

from libraries.Util import Util

from modelo.Model import Model

time.sleep(2)

print("* Starting audio classifier service...")

SERVER_REDIS = "localhost"  # settings.REDIS_HOST

# connect to Redis server
db = redis.StrictRedis(host=SERVER_REDIS,
                       port=settings.REDIS_PORT,
                       db=settings.REDIS_DB)

model = Model()
util = Util()


def classify_process():
    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0,
                          settings.BATCH_SIZE - 1)

        audioIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            audio = util.base64_to_audio(q["audio"])

            # check to see if the batch list is None
            if batch is None:
                batch = audio
            else:
                batch = np.vstack([batch, audio])

            audioIDs.append(q["id"])

        if len(audioIDs) > 0:
            batch = model.preparar_datos_para_modelo(batch)
            batch = np.array(batch)
            print("* Batch size: {}".format(batch.shape))

            results = model.predict(batch)
            for (audioID, resultSet) in zip(audioIDs, results[0]):
                output = []

                (predic, model_predic, _, vector, y_all) = resultSet

                r = {
                    "instruments_predict": predic,
                    "model_prediction": model_predic,
                    "All_instruments": y_all,
                    "time_predic": results[1]
                }
                output.append(r)
                db.set(audioID, json.dumps(output))
            db.ltrim(settings.IMAGE_QUEUE, len(audioIDs), -1)
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    data = util.preprocess_input("tmp/recibido.wav")
    data = np.array(data)
    print(data.shape)
    print(data)
    print("type", type(data))

    classify_process()
