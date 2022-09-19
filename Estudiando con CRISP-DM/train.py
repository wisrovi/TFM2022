import numpy as np
import pandas as pd
import pickle
from libraries.callbacks import get_callbacks
from libraries.network import plot_history
from libraries.optimizador import get_optimizador
from sklearn.model_selection import train_test_split

scaler = pickle.load(open('std_scaler.pkl', 'rb'))


def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = scaler.transform(np.array(train, dtype=float))
    y = instrument_list
    return X, y


"""

                        Cargando datos entrenamiento

"""
print("\n" * 5)
print("*" * 20)
print("Cargando datos")
test = pd.read_csv('train.csv')
X, y = prepare_data_tensorflow(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # 0.2
print(X_train.shape, y_train.shape)

"""

                        Preparando la red

"""
print("\n" * 5)
print("*" * 20)
print("Preparando la red")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, LeakyReLU, Conv1D, Conv2D, Flatten, \
    MaxPooling2D, Input
from tensorflow.keras.layers import BatchNormalization, InputLayer, Reshape, Activation, GlobalAveragePooling1D, \
    Normalization
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, UpSampling1D, UpSampling2D, MaxPooling1D

model = Sequential()
model.add(Dense(64, activation='relu',
                input_shape=(X_train.shape[1],)
                ))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

print(model.summary())

"""

                        Compilando el modelo

"""
print("\n" * 5)
print("*" * 20)
print("Compilando la red")
model.compile(optimizer=get_optimizador(),
              loss='mean_squared_error',  # categorical_crossentropy sparse_categorical_crossentropy
              metrics=['accuracy'])

"""

                        Entrenando el modelo

"""
print("\n" * 5)
print("*" * 20)
print("Entrenando la red")
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=500,
                    batch_size=32,
                    verbose=1,
                    callbacks=get_callbacks("primero")
                    )

plot_history(history)

"""

                        Guardando el modelo

"""
print("\n" * 5)
print("*" * 20)
print("Guardando el modelo")
name = "primero"
model.save("models/model_" + name + ".h5")

"""

                        Evaluar el modelo

"""
print("\n" * 5)
print("*" * 20)
print("Evaluando la red")
print("\n" * 5)
print("*" * 20)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Evaluate train acc: ', test_acc)

"""

                        Testear el modelo

"""
print("\n" * 5)
print("*" * 20)
print("Testeando la red")
test = pd.read_csv('test.csv')
X, y = prepare_data_tensorflow(test)

json_modelos = {
    "82_9": "model82_9.h5",
    "now": "models/model_primero.h5"
}

from tensorflow.keras.models import load_model

model = load_model(json_modelos['now'])

test_loss, test_acc = model.evaluate(X, y)

print('test_acc: ', test_acc)




################
################






