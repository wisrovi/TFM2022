import numpy as np

from libraries.Kaggle_audios import Kaggle_audios
from libraries.ProcessAudio import ProcessAudio

TIEMPO_SELECCIONADO = 2
ARCHIVO_FINAL_TRAIN = "data/scaler_pca_to_use/train"
ARCHIVO_FINAL_TEST = "data/scaler_pca_to_use/test"


def prepare_data_tensorflow(data):
    train = data[:, 1:-11]
    instrument_list = data[:,-11:]
    X = np.array(train, dtype=float)
    y = instrument_list
    return X, y


def aplicando_pca(scaler_pca, name_file, dataX, dataY):
    x_for_model = scaler_pca.transform(X=dataX)
    # print("Original", X.shape)
    np.savez_compressed(name_file + ".npz", data=x_for_model)
    np.savez_compressed(name_file + "_label.npz", data=dataY)


def leyendo_datos(name_file):
    x = np.load(name_file + ".npz")['data']
    y = np.load(name_file + "_label.npz")['data']
    return x, y


def Preprocesar_audios(name_file, use_train, save:bool = True):
    cortar = Kaggle_audios(config_time=TIEMPO_SELECCIONADO, train=use_train)

    print("\tLeyendo todos los archivos WAV originales")
    all_data, all_label, rate = cortar.read_data(
        limit=None, show_info=False)  # leer todos los wav y cada uno separarlos en pequeños audios de 3 segundos

    print("\tTime:", TIEMPO_SELECCIONADO, " - Input:", all_data.shape, " - Output:", all_label.shape, " - rate:", rate)

    print("\tExtrayendo caracteristicas audios")
    data = list()
    for id_audio, x in enumerate(all_data):
        processAudio = ProcessAudio()
        processAudio.set_data(x)
        data_save = processAudio.get_all(id_audio)  # Extrayendo caracteristicas audios, salen 26 caracteristicas
        data_save += all_label[id_audio].tolist()

        data.append(data_save)
        try:
            if id_audio%150 == 0:
                print("\n\t", end="")
            print(".", end="")
        except:
            print()

    print("\tConvirtiendo a numpy")
    data = np.array(data)

    if save:
        print("\tGuardando data")
        np.savez_compressed(name_file + '.npz', data)
        print("\tGuardando csv completo")

    return data










print("Preprocesando TRAIN")
data = Preprocesar_audios(ARCHIVO_FINAL_TRAIN, use_train=True, save=False)
print("\tTime:", TIEMPO_SELECCIONADO, " - Train:", len(data))



"""
            Hallando el Normalizador y el PCA (92%)
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X, _ = prepare_data_tensorflow(data)
MINIMA_VARIANA_EXPLICADA = 0.92
scaler_pca = make_pipeline(StandardScaler(), PCA(MINIMA_VARIANA_EXPLICADA))
scaler_pca.fit(X)

"""
            Aplicando PCA
"""


X, Y = prepare_data_tensorflow(data)
aplicando_pca(scaler_pca, ARCHIVO_FINAL_TRAIN, X.shape, Y )


print("Preprocesando TEST")
data = Preprocesar_audios(ARCHIVO_FINAL_TEST, False, save=False)
print("\tTime:", TIEMPO_SELECCIONADO, " - Test:", len(data))

X, Y = prepare_data_tensorflow(data)
aplicando_pca(scaler_pca, ARCHIVO_FINAL_TRAIN, X.shape, Y )


"""
            Leyendo datos
"""
xTrain_for_model_charge, yTrain = leyendo_datos(ARCHIVO_FINAL_TRAIN)
print("x_for_model_charge:", xTrain_for_model_charge.shape, yTrain.shape)

xTest_for_model_charge, yTest = leyendo_datos(ARCHIVO_FINAL_TEST)
print("x_for_model_charge:", xTest_for_model_charge.shape, yTest.shape)
