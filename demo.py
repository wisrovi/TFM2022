import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()

train = pd.read_csv('train.csv')
scaler.fit(np.array(train.iloc[:, 1:-11], dtype=float))


# https://stackoverflow.com/questions/53152627/saving-standardscaler-model-for-use-on-new-datasets
import pickle
pickle.dump(scaler, open('std_scaler.pkl','wb'))

def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    # print(instrument_list.head())

    train = data.iloc[:, 1:-11]  #
    # print(train.head())

    X = scaler.transform(np.array(train, dtype=float))
    y = instrument_list

    return X, y


X, y = prepare_data_tensorflow(train)

# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# y = encoder.fit_transform(instrument_list)
# print(y)


# print(X.shape, y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)

# Model

from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, LeakyReLU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import BatchNormalization, InputLayer, Reshape, Activation, GlobalAveragePooling1D
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, UpSampling1D, UpSampling2D, MaxPooling1D
def modelo_experimental(cantidad_entradas):
    #  https://github.com/wisrovi/MELI_test/blob/main/4_5_Modelado_Evaluacion_RNA.ipynb
    inputs = Input(shape=(cantidad_entradas,), name="Entradas")

    model = Sequential()
    model.add(inputs)

    model.add(Dense(16 * 2 * 2, activation="relu"))
    model.add(Reshape((2, 2, 16)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(UpSampling2D())

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

    model.add(Flatten())

    model.add(Dropout(0.25))  # apagar un 25% de manera aleatoria para reducir la cantidad de parametros

    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(cantidad_salidas, activation="sigmoid", name='output_layer'))

    return model


def modelo_simple():
    inputs = Input(shape=(cantidad_entradas,), name="Entradas")
    model = Sequential(name="Redsimple")  # los nombres van sin espacios
    model.add(inputs)
    model.add(Dense(4 * 4, activation="relu"))
    model.add(Reshape((4, 4)))
    model.add(Dense(32, activation="relu"))

    model.add(UpSampling1D(size=3))
    model.add(Conv1D(12, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(12, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(32, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))

    model.add(Conv1D(24, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(24, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(32, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))

    model.add(Conv1D(32, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(32, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))

    model.add(Conv1D(24, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(24, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(16, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))

    model.add(Conv1D(16, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(8, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))

    model.add(Dense(24, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))

    model.add(Dropout(0.5))

    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))

    # model.add( AveragePooling1D(pool_size=2, strides=1, padding='valid') )

    model.add(GlobalAveragePooling1D())

    model.add(Flatten())

    model.add(Dense(8, activation="relu"))
    model.add(Dense(3, activation="relu"))

    model.add(Dense(cantidad_salidas, activation="sigmoid", name='output_layer'))

    return model


model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(y_train.shape[1], activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',
              loss='mean_squared_error',  # categorical_crossentropy sparse_categorical_crossentropy
              metrics=['accuracy'])



from keras.callbacks import EarlyStopping

#EarlyStopping, detener el entrenamiento una vez que su pérdida comienza a aumentar
early_stop = EarlyStopping(
    monitor='accuracy',
    patience=5, #argumento de patience representa el número de épocas antes de detenerse una vez que su pérdida comienza a aumentar (deja de mejorar).
    min_delta=0,  #es un umbral para cuantificar una pérdida en alguna época como mejora o no. Si la diferencia de pérdida es inferior a min_delta , se cuantifica como no mejora. Es mejor dejarlo como 0 ya que estamos interesados ​​en cuando la pérdida empeora.
    mode='auto')
callbacks_list = [early_stop]

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    verbose=1,
                    callbacks=callbacks_list
                    )

print("*" * 20)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Evaluate train acc: ', test_acc)

test = pd.read_csv('test.csv')
X, y = prepare_data_tensorflow(test)

print("*" * 20)
test_loss, test_acc = model.evaluate(X, y)
print('test_acc: ', test_acc)

model.save("model.h5")


