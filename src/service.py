from flask import Flask, app, jsonify, request, redirect, make_response
from leer_modelo import predecir

ALLOWED = ['png','jpg', 'jpeg', 'gif']
def evaluar_extension_archivo(filename):
    tiene_punto = "." in filename
    if tiene_punto:
        extension_archivo = filename.split(".", 1)[1].lower()
        if extension_archivo  in ALLOWED:
            return True
    return False

nombres_parametros = {
    "imagen":"file1"
}


html = """
<!doctype html>
<form method="POST" enctype="multipart/form-data">
  
  <label for="fname">Elija su imagen a evaluar:</label>
  <input type="file" id="fname" name="file1"><br><br>
  
  <input type="submit" value="Evaluar con RNA">
</form> 
"""

nombre_guardar_archivo = "recibido.jpg"

app = Flask(__name__)

@app.route("/RNA", methods=["POST", "GET"])
def recibir_archivo():
    if request.method == "POST":
        if nombres_parametros["imagen"] not in request.files:
            redirect(request.url)
            
        nombre_imagen_recibida = request.files["file1"]
        if nombre_imagen_recibida.filename == "":
            redirect(request.url)
            
        if evaluar_extension_archivo(nombre_imagen_recibida.filename):
            nombre_imagen_recibida.save(nombre_guardar_archivo)
            
            # evaluacion por el modelo de RNA
            rta = predecir(nombre_guardar_archivo)
            return "La imagen recibida es un " + rta
    
    return html


if __name__=="__main__":
    #print(evaluar_extension_archivo("queso.jpg"))
    app.run(host="0.0.0.0", port=2022, debug=True)
