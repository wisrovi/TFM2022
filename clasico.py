import pickle
scaler = pickle.load(open('std_scaler.pkl', 'rb'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = scaler.transform(np.array(train, dtype=float))
    y = instrument_list
    return X, y

print("Cargando datos")
test = pd.read_csv('train.csv')
X, y = prepare_data_tensorflow(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # 0.2
print(X_train.shape, y_train.shape)


from sklearn.ensemble import RandomForestClassifier
algoritmo_5 = RandomForestClassifier()

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
seed = 1
grid = GridSearchCV(
          estimator = algoritmo_5,
          param_grid={},
          cv = KFold(n_splits=10, shuffle=True, random_state=seed)
        )

"""

                        Entrenando el modelo

"""
grid.fit(X_train, y_train)
algoritmo_5 = grid.best_estimator_

y_final = algoritmo_5.predict(X_test)

import sklearn.metrics as metrics
print("ACC", metrics.accuracy_score(y_test, y_final))
print("PREC", metrics.precision_score(y_test, y_final, average='micro'))


pickle.dump(algoritmo_5, open('randomforest.pkl','wb'))