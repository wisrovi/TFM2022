import pickle
from functools import wraps
from time import time
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
        if not self.data_good:
            if len(data) == 26:
                data = [data]
            data = self.preparar_datos_para_modelo(data)

        y_final = model.predict(data)[0]
        y_final_h = util.decode_prediction(y_final)
        y_final = [int(pred) for pred in y_final]
        y_final = [True if pred == 1 else False for pred in y_final]

        predicciones = model.predict_proba(data)

        self.data_good = False

        return y_final_h, y_final, predicciones, data, util.y_all

    def preparar_datos_para_modelo(self, datos):
        x_for_model = normalizador_pca.transform(X=datos)
        self.data_good = True
        return x_for_model
