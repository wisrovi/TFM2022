import uuid
import time
import json
import os
from fastapi import UploadFile, File
from fastapi.templating import Jinja2Templates

from libraries.Util_received_file import evaluar_extension_archivo
from libraries.base_api import util, db
from libraries.api_template.base import app, Request

from config import settings

templates = Jinja2Templates(directory="templates")


@app.get("/info/")
async def version():
    return "Esta es una prueba usando el template."


@app.get("/RNA/")
async def version(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/RNA/")
async def create_upload_file(file1: UploadFile = File(...), test: str = None):
    if evaluar_extension_archivo(file1.filename) or test is not None:
        # generate an ID for the classification then add the
        # classification ID + audio to the queue of cluster
        k = str(uuid.uuid4())
        nombre_guardar_archivo = f"tmp/{k}.wav"
        try:
            contents = file1.file.read()
            with open(nombre_guardar_archivo, 'wb') as f:
                f.write(contents)
        except Exception as e:
            return {"message": "There was an error uploading the file"}
        finally:
            file1.file.close()

        # extraer caracteristicas
        data = util.preprocess_input(nombre_guardar_archivo)

        # borrar archivo si existe
        if os.path.exists(nombre_guardar_archivo):
            os.remove(nombre_guardar_archivo)

        # guardar en redis
        data = util.audio_to_base64(data)
        d = {"id": k, "audio": data}
        db.rpush(settings.NAME_QUEUE, json.dumps(d))

        # print del nombre de la variable de entorno llamada CLUSTER_NAME
        print(os.environ.get("CLUSTER_NAME", "cluster1"))

        rta = {}
        while True:
            output = db.get(k)
            if output is not None:
                output = output.decode("utf-8")
                rta = json.loads(output)
                db.delete(k)
                break
            time.sleep(settings.CLIENT_SLEEP)
        if len(rta) > 0:
            print(rta[0].get("instruments_predict", "file: " + nombre_guardar_archivo))

        return rta

    return {"error": f"la extension del archivo no es correcta, el archivo recibido fue: {file1.filename}"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


# para ejecutar localmente en Debug
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2022)
