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




import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

# Métricas de evaluación.
metricas_exito = {
  'ACC':    metrics.accuracy_score,                                                          # Exactitud
  'PREC':   lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='micro'), # Precisión
  'RECALL': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='micro'),    # Sensibilidad
  'F1':     lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='micro'),        # F1
  'Matriz_confusion' : metrics.confusion_matrix,
  'Tabla_metricas' : metrics.classification_report
}

from math import sqrt

# Métricas de evaluación.
metricas_error = {
  'MSE':  metrics.mean_squared_error,
  'MAE':  metrics.mean_absolute_error,
  'RMSE': lambda y, y_pred:
          sqrt(metrics.mean_squared_error(y, y_pred)),
  'MAPE': lambda y, y_pred:
          np.mean(np.abs((y - y_pred) / y)) * 100,
  'R2':   metrics.r2_score
  }

algoritmo_1 = LogisticRegression(
          solver='sag', #'liblinear', 'sag'
          max_iter=1000,
          random_state=1, #0 ,1
          multi_class='ovr'
          )
# El árbol se crece al máximo posible para luego aplicar el pruning
algoritmo_2 = DecisionTreeClassifier(
          max_depth = None,
          min_samples_split = 2,
          min_samples_leaf  = 1,
          random_state      = 123
          )
algoritmo_3 = SVC(kernel="rbf")
algoritmo_4 = DummyClassifier(strategy='stratified')
algoritmo_5 = RandomForestClassifier()
algoritmo_6 = GaussianNB()
algoritmo_7 = KNeighborsClassifier(7)

# Construcción del algoritmo de aprendizaje.
algoritmos = {
    'DummyClassifier': algoritmo_4,
    'SVM': algoritmo_3,
    'LOGR': algoritmo_1,
    'TREE': algoritmo_2,
    'RandomForest': algoritmo_5,
    'GaussianNaiveBayes': algoritmo_6,
    'KNeighbors_knn': algoritmo_7
}

parametros_grid = {
    'LOGR': {
    },
    'GaussianNaiveBayes': {
    },
    'KNeighbors_knn': {
    },
    'SVM': {
        "C": [1, 10, 100],
        "gamma": [.01, .1]
    },
    'TREE': {
        'ccp_alpha': np.linspace(0, 80, 20)
    },
    'DummyClassifier': {
    },
    'RandomForest': {
    },
}
# creamos una bodega de almacenamiento de todos los modelos entrenados

y_pred = dict()
for nombre, alg in algoritmos.items():
  y_pred[nombre] = dict()

seed = 1

for nombre, alg in algoritmos.items():
    print("Entrenando usando:", nombre)
    grid = GridSearchCV(
        estimator=alg,
        param_grid=parametros_grid[nombre],
        cv=KFold(n_splits=10, shuffle=True, random_state=seed)
    )

    train_ok = False

    try:
        # print(len(X_train), len(y_train))
        grid.fit(X_train, y_train)
        # print(grid.best_params_)
        train_ok = True
    except Exception as e:
        print(str(e)[str(e).find("ValueError")])

    if train_ok:
        y_pred[nombre]['model'] = grid.best_estimator_

        pickle.dump(y_pred[nombre]['model'] , open('models_backup/' + nombre + '.pkl', 'wb'))

for nombre, alg in algoritmos.items():
    y_final = y_pred[nombre]['model'].predict(X_test)

    # Cálculo de las métricas de evaluación.
    exito = dict()
    for key, metrica_usar in metricas_exito.items():
        exito[key] = metrica_usar(y_test, y_final)

    print(nombre)
    print("ACC", metrics.accuracy_score(y_test, y_final))
    print("PREC", metrics.precision_score(y_test, y_final, average='micro'))
    print("PREC", metrics.precision_score(y_test, y_final, average='macro'))
    print("PREC", metrics.precision_score(y_test, y_final, average='weighted'))
    print("F1", metrics.f1_score(y_test, y_final, average='micro'))

    resultados = metrics.classification_report(y_test, y_final)
    print(type(resultados))
    print(resultados)
    y_pred[nombre]['exito'] = exito

    errores = dict()
    for key, metrica_usar in metricas_error.items():
        errores[key] = metrica_usar(y_test, y_final)

    y_pred[nombre]['error'] = errores

modelo = dict()

for nombre, alg in algoritmos.items():
  if y_pred[nombre]['error']['R2'] >= 0:
    if y_pred[nombre]['error']['R2'] == 1:
      pass
    print(nombre)
    print("ACC", y_pred[nombre]['exito']['ACC'] )
    print("PREC", y_pred[nombre]['exito']['PREC'] )

    print('MSE', y_pred[nombre]['error']['MSE'])

  print()

