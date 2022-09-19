# Este archivo se basa en la desicion tomada segun lo explicado en la imagen: "Razon_Eleccion_3seg_PCA92%.jpg"

import pandas as pd
import numpy as np
import pickle

def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = np.array(train, dtype=float)
    y = instrument_list
    return X, y


test = pd.read_csv("data/train_3seg.csv")
X, y = prepare_data_tensorflow(test)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MINIMA_VARIANA_EXPLICADA = 0.92
pca_pipe = make_pipeline(StandardScaler(), PCA(MINIMA_VARIANA_EXPLICADA))
pca_pipe.fit(X)
pickle.dump(pca_pipe, open('scaler_pca.pkl','wb'))


scaler_pca = pickle.load(open('scaler_pca.pkl', 'rb'))
x_for_model = scaler_pca.transform(X=X)
print(x_for_model.shape)