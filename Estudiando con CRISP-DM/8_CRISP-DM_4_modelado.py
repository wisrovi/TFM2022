import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

ARCHIVO_FINAL_TRAIN = "data/scaler_pca_to_use/train"
ARCHIVO_FINAL_TEST = "data/scaler_pca_to_use/test"


def leyendo_datos(name_file):
    x = np.load(name_file + ".npz")['data']
    y = np.load(name_file + "_label.npz")['data']
    return x, y


mean = lambda lst: int((sum(lst) / len(lst)) * 100) / 100


def calcular_porcentajes_aciertos(y_f, y_t):
    verdaderos = dict()
    falsos = dict()
    for j in range(y_f.shape[1]):
        verdaderos[j] = 0
        falsos[j] = 0

    for i in range(y_f.shape[0]):
        for j in range(y_f.shape[1]):
            if y_f[i][j] == y_t[i][j]:
                verdaderos[j] += 1
            else:
                falsos[j] += 1

    for j in range(y_f.shape[1]):
        # y_final.shape[1] -> 100%
        # verdaderos[j]    -> X
        verdaderos[j] = int(verdaderos[j] * 100 / y_f.shape[0])
        falsos[j] = int(falsos[j] * 100 / y_f.shape[0])

    return verdaderos, falsos, str(mean([v for i, v in verdaderos.items()])) + "%"


"""
            Leyendo datos
"""
xTrain_for_model_charge, yTrain = leyendo_datos(ARCHIVO_FINAL_TRAIN)
yTrain = np.array(yTrain, dtype=float)
xTrain_for_model_charge = np.array(xTrain_for_model_charge, dtype=float)
print("datos train:", xTrain_for_model_charge.shape, yTrain.shape)


xTest_for_model_charge, yTest = leyendo_datos(ARCHIVO_FINAL_TEST)
yTest = np.array(yTest, dtype=float)
xTest_for_model_charge = np.array(xTest_for_model_charge, dtype=float)
print("datos test:", xTest_for_model_charge.shape, yTest.shape)

# usando un modelo clasico randomFest


"""
            Usan un modelo tradicional
"""
#model = pickle.load(open('data/scaler_pca_to_use/randomforest.pkl', 'rb'))
#y_final = model.predict(xTest_for_model_charge)
#aciertos = calcular_porcentajes_aciertos(y_final, yTest)[2]
#print("[randomforest]: Aciertos test", aciertos)


"""
            Entrenando un RNA
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, LeakyReLU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import BatchNormalization, InputLayer, Reshape, Activation, GlobalAveragePooling1D
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, UpSampling1D, UpSampling2D, MaxPooling1D

from tensorflow.keras.optimizers import Adam
def get_optimizador():
    adam = Adam(learning_rate=1e-5)
    return adam


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, RemoteMonitor, TerminateOnNaN, \
    BackupAndRestore


def get_callbacks(name="model"):
    # EarlyStopping, detener el entrenamiento una vez que su pérdida comienza a aumentar
    early_stop = EarlyStopping(
        monitor='accuracy',
        patience=8,
        # argumento de patience representa el número de épocas antes de detenerse una vez que su pérdida comienza a aumentar (deja de mejorar).
        min_delta=0,
        # es un umbral para cuantificar una pérdida en alguna época como mejora o no. Si la diferencia de pérdida es inferior a min_delta , se cuantifica como no mejora. Es mejor dejarlo como 0 ya que estamos interesados ​​en cuando la pérdida empeora.
        restore_best_weights=True,
        mode='max')

    # ReduceLROnPlateau, que si el entrenamiento no mejora tras unos epochs específicos, reduce el valor de learning rate del modelo
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=5,
        min_delta=1e-4,
        mode='min',
        verbose=1,
    )

    # Saves Keras model after each epoch
    # Para algunos casos es importante saber cual entrenamiento fue mejor,
    # este callback guarda el modelo tras cada epoca completada con el fin de si luego se desea un registro de pesos para cada epoca
    # Se ha usado este callback para poder optener el mejor modelo de pesos, sobretodo en la red neuronal creada desde cero
    # siendo de gran utilidad para determinar el como ir modificando los layer hasta obtener el mejor modelo
    checkpointer = ModelCheckpoint(
        filepath='models_backup/' + name + '-{val_accuracy:.4f}.h5',
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True,
        save_weights_only=False
    )

    remote_monitor = RemoteMonitor(
        root='http://localhost:6006',
        path='/publish/epoch/end/',
        field='data',
        headers=None,
        send_as_json=False
    )

    backup_restore = BackupAndRestore(backup_dir="backup")

    proteccion_nan_loss = TerminateOnNaN()

    callbacks_list = [early_stop, reduce_lr, proteccion_nan_loss, backup_restore]  #, checkpointer , remote_monitor]

    return callbacks_list


import matplotlib.pyplot as plt
def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

cantidad_entradas = 15
cantidad_salidas = 11

model = Sequential(name="RedBasica")
model.add(Dense(32, activation='relu', input_shape=(cantidad_entradas,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(cantidad_salidas, activation='softmax', name='output_layer'))

model.summary()

model.compile(optimizer=get_optimizador(),
              loss='mse',  # categorical_crossentropy sparse_categorical_crossentropy mean_squared_error
              metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(xTrain_for_model_charge, yTrain, test_size=0.1)  # 0.2




history = model.fit(
    X_train,
    y_train,
    #validation_data=(X_test, y_test),
    epochs=500,
    #batch_size=64,
    verbose=1,
    #callbacks=get_callbacks()
)