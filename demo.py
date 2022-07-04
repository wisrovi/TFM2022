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

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
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


