import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("modelo/model.h5")

def convertir_imagen_para_model(nombre_archivo):
    img = image.load_img(nombre_archivo, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])
    return x

def predecir(nombre_archivo):
    imagen = convertir_imagen_para_model(nombre_archivo)
    classes = model.predict(imagen, batch_size=10)
    #print(classes)
    if classes[0] > 0.5:
        return "DOG"
    else:
        return "CAT"
    
if __name__=="__main__":
    rta = predecir("recibido.jpg")
    print(rta)
