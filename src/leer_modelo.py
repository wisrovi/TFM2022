import pickle
from ProcessAudio import ProcessAudio


normalizador_pca = pickle.load(open('modelo/normalizador_pca_1seg.pkl', 'rb'))
model = pickle.load(open("modelo/randomforest.pkl", 'rb'))


from functools import wraps
from time import time


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


def preparar_datos_para_modelo(datos):    
    x_for_model = normalizador_pca.transform(X=datos)
    return x_for_model


def preparar_dato(file_path):
    processAudio = ProcessAudio()
    processAudio.set_data(file_path)
    DATA = processAudio.get_all()  # Extrayendo caracteristicas audios, salen 26 caracteristicas
    return DATA


@count_elapsed_time
def predecir(file_path):
    DATA = preparar_dato(file_path)    # Extrayendo caracteristicas audios, salen 26 caracteristicas
    DATA_REDU = preparar_datos_para_modelo([DATA])
    
    y_final, predicciones = None , None    
    y_final = model.predict([DATA_REDU[0]])
    #predicciones = model.predict_proba(DATA_REDU)
    return y_final, predicciones, DATA_REDU

    
if __name__=="__main__":
    rta = predecir("demo/dat_1.wav")
    print(rta)
