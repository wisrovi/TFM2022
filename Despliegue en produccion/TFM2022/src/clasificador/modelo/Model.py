import pickle
from functools import wraps
from time import time
import numpy as np
from libraries.Util import Util

util = Util()

model = pickle.load(open("modelo/randomforest.pkl", 'rb'))
normalizador_pca = pickle.load(open('modelo/normalizador_pca_1seg.pkl', 'rb'))


def count_elapsed_time(f):
    @wraps(f)
    def cronometro(*args, **kwargs):
        t_inicial = time()  # tomo la hora antes de ejecutar la funcion
        salida = f(*args, **kwargs)
        t_final = time()  # tomo la hora despues de ejecutar la funcion
        duracion_segundos = t_final - t_inicial
        print('Tiempo transcurrido (en segundos): {}'.format(duracion_segundos))
        return salida, duracion_segundos

    return cronometro


class Model:
    data_good = False

    def __init__(self):
        pass

    @count_elapsed_time
    def predict(self, data):
        y_final = model.predict(data)
        #predicciones = model.predict_proba(data)

        # decodificar y_final con decode_prediction
        nuevas_f_h = []
        nuevas_y_f = []
        for i, y_f in enumerate(y_final):
            y_final_h = util.decode_prediction(y_f)
            nuevas_f_h.append(y_final_h)

            y_f = [int(pred) for pred in y_f]
            y_f = [True if pred == 1 else False for pred in y_f]
            nuevas_y_f.append(y_f)

        y_final_h = nuevas_f_h
        y_final = nuevas_y_f

        self.data_good = False

        return y_final_h, y_final, [None for _ in range(len(y_final_h))], data, util.y_all

    def preparar_datos_para_modelo(self, datos, size=None):
        if size is not None:
            if size == 1:
                datos = [datos]
        x_for_model = normalizador_pca.transform(X=datos)
        self.data_good = True
        x_for_model = np.array(x_for_model)
        return x_for_model
