import json
import numpy as np
import config.settings as settings
from functools import wraps
import time
import redis

from libraries.Util import Util
from libraries.Log import logging
from modelo.Model import Model

time.sleep(2)

print("* Starting audio classifier service...")  # imprimo mensaje de inicio
logging.info("* Starting audio classifier service...")  # inicio el servicio

# connect to Redis server
try:
    db = redis.StrictRedis(host=settings.REDIS_HOST,
                           port=settings.REDIS_PORT,
                           db=settings.REDIS_DB)
except Exception as e:
    logging.error("Error en conexión a redis: {}".format(e))  # muestro error en conexión a redis
    exit()  # salgo del programa


model = Model()
util = Util()


def organizar_respuesta_modelo(results):
    tiempo = results[1]
    results = results[0]
    instrumentos = results[4]
    results = results[:4]

    resultados_para_humanos = results[0]
    resultados_para_modelo = results[1]
    originales = results[3]

    new_results = []
    for i, r in enumerate(resultados_para_humanos):
        new_results.append((resultados_para_humanos[i], resultados_para_modelo[i], originales[i]))
    results = new_results

    return tiempo, instrumentos, results


def organizar_json_respuesta_individual(resultSet, tiempo, instrumentos):
    (respuesta_humana, model_predic, _) = resultSet

    r = {
        "instruments_predict": respuesta_humana,
        "model_prediction": model_predic,
        "All_instruments": instrumentos,
        "time_predic": tiempo
    }
    return r


def classify_process():
    while True:
        try:
            queue = db.lrange(settings.NAME_QUEUE, 0,
                              settings.BATCH_SIZE - 1)  # reviso en redis la lista de peticiones
        except Exception as e:
            logging.error("Error en redis: {}".format(e))  # si hay error, lo muestro y sigo
            continue  # si hay error, salto a la siguiente iteracion del while

        audioIDs = []  # limpio la lista de ids
        batch = None  # limpio la lista del lote a procesar

        for q in queue:  # recorro la lista de peticiones y para cada peticion extraigo los datos a procesar
            q = json.loads(q.decode("utf-8"))  # extraigo la peticion
            audio = util.base64_to_audio(q["audio"])  # extraigo el dato
            batch = audio if batch is None else np.vstack(
                [batch, audio])  # lleno un numpy array con los datos recibidos
            audioIDs.append(q["id"])  # lleno una lista con los ids recibidos

        if len(audioIDs) > 0:  # si hay al menos una peticion para que el modelo procese la proceso de lo contrario hago una pausa y vuelvo a revisar
            try:
                batch = model.preparar_datos_para_modelo(batch, size=len(
                    audioIDs))  # hago pre-procesamiento de datos para que el modelo pueda predecir
            except Exception as e:
                logging.error(
                    "Error en pre-procesamiento de datos para modelo: {}".format(e))  # si hay error, lo muestro y sigo
                batch = None  # si hay error, salto a la siguiente iteracion del while
                continue

            print("* Batch size: {}".format(batch.shape))  # imprimo el tamaño del lote a procesar

            try:
                results = model.predict(batch)  # el modelo realiza sus predicciones
            except Exception as e:
                logging.error("Error en predicción de modelo: {}".format(e))  # si hay error, lo muestro y sigo
                results = None  # si hay error, salto a la siguiente iteracion del while
                continue

            try:
                tiempo, instrumentos, results = organizar_respuesta_modelo(
                    results)  # la respuesta del modelo la convierto en datos legibles
            except Exception as e:
                logging.error(
                    "Error en organización de respuesta de modelo para organizar la respuesta del usuario: {}".format(
                        e))  # si hay error, lo muestro y sigo
                results = None  # si hay error, salto a la siguiente iteracion del while
                continue

            logging.debug("* Batch size: {} - time: {} segundos".format(batch.shape, round(tiempo, 3)))  # imprimo el tamaño del lote a procesar

            for (audioID, resultSet) in zip(audioIDs,
                                            results):  # para cada respuesta del modelo genero un json de respuesta al usuario
                try:
                    output = []  # limpio la lista de respuestas
                    r = organizar_json_respuesta_individual(resultSet, tiempo,
                                                            instrumentos)  # creo el json de respuesta para cada peticion
                    output.append(r)  # agrego el json a la lista de respuestas
                    db.set(audioID,
                           json.dumps(output))  # pongo el json del usuario en redis con llave el id del dato recibido
                except Exception as e:
                    logging.error(
                        "Error en preparar el json de respuesta: {}".format(e))  # si hay error, lo muestro y sigo
                    continue
            try:
                db.ltrim(settings.NAME_QUEUE, len(audioIDs),
                         -1)  # luego de procesar la lista de peticiones borro el lote procesado
            except Exception as e:
                logging.error(
                    "Error en limpiar la lista de peticiones: {}".format(e))  # si hay error, lo muestro y sigo
                continue
        time.sleep(
            settings.SERVER_SLEEP)  # hago una pausa en espera de que la lista de peticiones pueda llenarse con una o mas peticiones a procesar


if __name__ == "__main__":
    classify_process()  # ejecuto el proceso de clasificacion
