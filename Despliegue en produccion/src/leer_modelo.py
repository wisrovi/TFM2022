import pickle
import pandas as pd
from ProcessAudio.Features import Features
from functools import wraps
from time import time


instrumentos = {
        1: "piano",
        7: "Violin",
        41: "Viola",
        42: "Violonchelo",
        43: "Clarinete",
        44: "Fagot",
        61: "Bocina",
        69: "Oboe",
        71: "Flauta",
        72: "Clave",
        74: "Contrabajo",
    }  


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


def read_data_label():
    TODOS_LABEL = pd.read_csv("modelo/all_label.csv", header=0)
    TODOS_LABEL = TODOS_LABEL.to_numpy()
    TODOS_LABEL = TODOS_LABEL.tolist()
    TODOS_LABEL = tuple([ d[1] for d in TODOS_LABEL])
    return TODOS_LABEL


def preparar_datos_para_modelo(datos):    
    x_for_model = normalizador_pca.transform(X=datos)
    return x_for_model


def preparar_dato(file_path):
    processAudio = Features()
    processAudio.set_data(file_path)
    DATA = processAudio.build_all()  # Extrayendo caracteristicas audios, salen 26 caracteristicas
    return DATA


def decode_prediction(prediction):
    REAL_PREDICTION = []
    for i, p in enumerate(prediction):
        p = int(p)
        if p == 1:
            usado = TODOS_LABEL[i]
            REAL_PREDICTION.append(f"{instrumentos[usado]}({usado})")
    return REAL_PREDICTION


@count_elapsed_time
def predecir(file_path):
    DATA = preparar_dato(file_path)    # Extrayendo caracteristicas audios, salen 26 caracteristicas
    DATA_REDU = preparar_datos_para_modelo([DATA]) # estandarizo y reduzco dimensiones con PCA
    #DATA_REDU = DATA_REDU[0] # saco el array de dentro del array
    y_final = model.predict(DATA_REDU)[0]
    predicciones = model.predict_proba(DATA_REDU)
    y_final_h = decode_prediction(y_final)
    y_final = [int(pred) for pred in y_final]
    y_final = [True if pred == 1 else False for pred in y_final]    
    return (y_final_h, y_final, predicciones, DATA_REDU, y_all)


normalizador_pca = pickle.load(open('modelo/normalizador_pca_1seg.pkl', 'rb'))
model = pickle.load(open("modelo/randomforest.pkl", 'rb'))
TODOS_LABEL = read_data_label()
print("Todo listo para predecir a las salidas:", TODOS_LABEL)
y_all = decode_prediction([1 for _ in range(len(instrumentos))])

    
if __name__=="__main__":
    print("Probando modelo")
    (predic, _, vector), tiempo = predecir("demo/dat_92.wav")
    
    print(predic)
    print(vector)
    print(tiempo)
