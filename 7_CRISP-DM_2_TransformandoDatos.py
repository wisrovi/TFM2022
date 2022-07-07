import pandas as pd
import numpy as np
import csv
import pickle

from libraries.Kaggle_audios import Kaggle_audios
from libraries.ProcessAudio import ProcessAudio

TIEMPO_SELECCIONADO = 3
ARCHIVO_FINAL_TRAIN = "data/scaler_pca_to_use/train"
ARCHIVO_FINAL_TEST = "data/scaler_pca_to_use/test"


def Preprocesar_audios(name_file, use_train):
    cortar = Kaggle_audios(config_time=TIEMPO_SELECCIONADO, train=use_train)

    print("\tLeyendo todos los archivos WAV originales")
    all_data, all_label, rate = cortar.read_data(
        limit=None, show_info=False)  # leer todos los wav y cada uno separarlos en pequeños audios de 3 segundos

    print("\tTime:", TIEMPO_SELECCIONADO, " - Input:", all_data.shape, " - Output:", all_label.shape, " - rate:", rate)

    print("\tExtrayendo caracteristicas audios")
    for id_audio, x in enumerate(all_data):
        processAudio = ProcessAudio()
        processAudio.set_data(x)
        data_save = processAudio.get_all(id_audio)  # Extrayendo caracteristicas audios, salen 26 caracteristicas
        data_save += all_label[id_audio].tolist()

        print("\tGuardando csv")

        with open(name_file + ".csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_save)

print("Preprocesando TRAIN")
Preprocesar_audios(ARCHIVO_FINAL_TRAIN, True)
print("Preprocesando TEST")
Preprocesar_audios(ARCHIVO_FINAL_TEST, False)

"""
            Aplicando PCA
"""

scaler_pca = pickle.load(open('scaler_pca.pkl', 'rb'))


def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = np.array(train, dtype=float)
    y = instrument_list
    return X, y


def aplicando_pca(name_file):
    data = pd.read_csv(name_file)
    X, y = prepare_data_tensorflow(data)

    x_for_model = scaler_pca.transform(X=X)
    print("Original", X.shape)
    np.savez_compressed(name_file + ".npz", data=x_for_model)
    np.savez_compressed(name_file + "_label.npz", data=y)


def leyendo_datos(name_file):
    x = np.load(ARCHIVO_FINAL_TRAIN + ".npz")['data']
    y = np.load(ARCHIVO_FINAL_TRAIN + "_label.npz")['data']
    return x, y


aplicando_pca(ARCHIVO_FINAL_TRAIN)
aplicando_pca(ARCHIVO_FINAL_TEST)

xTrain_for_model_charge, yTrain = leyendo_datos(ARCHIVO_FINAL_TRAIN)
print("x_for_model_charge:", xTrain_for_model_charge.shape, yTrain.shape)

xTest_for_model_charge, yTest = leyendo_datos(ARCHIVO_FINAL_TRAIN)
print("x_for_model_charge:", xTest_for_model_charge.shape, yTest.shape)
