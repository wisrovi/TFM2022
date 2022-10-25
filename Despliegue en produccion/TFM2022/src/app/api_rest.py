import json
import os
import redis
from flask import Flask, app, jsonify, request, redirect, make_response, render_template
import uuid
from libraries.Util import Util
from libraries.Util_received_file import evaluar_extension_archivo
#from modelo.Model import Model
from config import settings
import time


util = Util()


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
        test = request.values.get('test')

        if nombres_parametros["imagen"] not in request.files:
            redirect(request.url)

        nombre_imagen_recibida = request.files[nombres_parametros["imagen"]]
        if nombre_imagen_recibida.filename == "":
            redirect(request.url)

        if evaluar_extension_archivo(nombre_imagen_recibida.filename) or test is not None:
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())

            # guardar archivo de forma temporal
            nombre_guardar_archivo = f"tmp/{k}.wav"
            nombre_imagen_recibida.save(nombre_guardar_archivo)

            # extraer caracteristicas
            data = util.preprocess_input(nombre_guardar_archivo)
            
            #borrar archivo si existe
            if os.path.exists(nombre_guardar_archivo):
                os.remove(nombre_guardar_archivo)
            
            # guardar en redis
            data = util.audio_to_base64(data)
            d = {"id": k, "audio": data}
            # print("data", d)
            db.rpush(settings.NAME_QUEUE, json.dumps(d))

            # print del nombre de la variable de entorno llamada CLUSTER_NAME
            print(os.environ.get("CLUSTER_NAME"))

            # espera la respuesta del modelo
            rta = {}
            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    rta = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)
            print(rta[0].get("instruments_predict", "file: " + nombre_guardar_archivo))
            return jsonify(rta)
        else:
            print(test)
            return jsonify({"error": f"la extension del archivo no es correcta, el archivo recibido fue: {nombre_imagen_recibida.filename}" })

    # return html
    return render_template('index.html', **locals())


if __name__ == "__main__":
    # print(evaluar_extension_archivo("queso.jpg"))
    app.run(host="0.0.0.0", port=2022, debug=True)
