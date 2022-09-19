import pandas as pd
import numpy as np
import os
import pickle
#scaler = pickle.load(open('std_scaler.pkl', 'rb'))


def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = np.array(train, dtype=float)
    y = instrument_list
    return X, y


base_path = os.getcwd().split("musicnet")[0] + "musicnet/tf/data/"
files = os.listdir(base_path)
files.sort()


result_train = dict()
for file in files:
    if file.find("train")>=0:
        test = pd.read_csv(base_path+file)
        X, y = prepare_data_tensorflow(test)
        result_train[int(file.split("_")[1].split("seg")[0])] = {
            "file":file,
            "shape": (X.shape, y.shape),
            "data" : X
        }

result_train = dict(sorted(result_train.items()))



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


import pandas as pd
import numpy as np
temporal = data = pd.read_csv("cantidad_datos_segun_split_segundos.csv", index_col=0)
data = pd.read_csv("cantidad_entradas_segun_split_segundos.csv", index_col=0)
data["Cantidad_datos_train"] = temporal['Cantidad_datos_train']
data["Cantidad_datos_test"] = temporal['Cantidad_datos_test']
data["Entradas_luego_extraer_caracteristicas"] = [26 for _ in data["Division_segundos"] ]




for buscar in range(85, 100, 1):
    MINIMA_EXPLICACION_BUSCADA = buscar/100

    PCA_resumen = list()
    for key, value in result_train.items():
        pca_pipe = make_pipeline(StandardScaler(), PCA())
        pca_pipe.fit(result_train[key]['data'])
        modelo_pca = pca_pipe.named_steps['pca']
        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()

        componentes_recomendado = None
        for i, valu in enumerate(prop_varianza_acum):
            if valu > MINIMA_EXPLICACION_BUSCADA:
                componentes_recomendado = i + 1
                break
        PCA_resumen.append(componentes_recomendado)
        #print(f"El split de tiempo {key} se puede reducir minimo a {componentes_recomendado} para garantizar mantener el {int(MINIMA_EXPLICACION_BUSCADA*100)}% de la informacion original")



    data[f"Entradas_luego_PCA({int(MINIMA_EXPLICACION_BUSCADA*100)}%)"] = PCA_resumen



data.to_csv("Resumen_transformaciones_datos.csv")

print(data)