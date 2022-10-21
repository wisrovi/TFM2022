import json

import redis
from flask import Flask, app, jsonify, request, redirect, make_response, render_template
import uuid
from libraries.Util import Util
from libraries.Util_received_file import evaluar_extension_archivo
#from modelo.Model import Model
from config import settings
import time

util = Util()
#model = Model()

nombres_parametros = {
    "imagen": "file1"
}

nombre_guardar_archivo = "tmp/recibido.wav"

app = Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT,
                       db=settings.REDIS_DB)


@app.route("/RNA", methods=["POST", "GET"])
def recibir_archivo():
    if request.method == "POST":
        if nombres_parametros["imagen"] not in request.files:
            redirect(request.url)

        nombre_imagen_recibida = request.files[nombres_parametros["imagen"]]
        if nombre_imagen_recibida.filename == "":
            redirect(request.url)

        if evaluar_extension_archivo(nombre_imagen_recibida.filename):
            nombre_imagen_recibida.save(nombre_guardar_archivo)

            data = util.preprocess_input(nombre_guardar_archivo)

            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            data = util.audio_to_base64(data)
            d = {"id": k, "audio": data}
            print("data", d)
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            rta = {}
            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    rta = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)

            # evaluacion por el modelo de RNA
            #(predic, model_predic, _, vector, y_all), tiempo = model.predict(data)

            #rta = {
            #    "instruments_predict": predic,
            #    "model_prediction": model_predic,
            #    "All_instruments": y_all,
            #    "time_predic": tiempo
            #}
            # return rta
            return jsonify(rta)

    # return html
    return render_template('index.html', **locals())


if __name__ == "__main__":
    # print(evaluar_extension_archivo("queso.jpg"))
    app.run(host="0.0.0.0", port=2022, debug=True)
