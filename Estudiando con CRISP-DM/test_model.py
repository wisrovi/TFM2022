import numpy as np
import pandas as pd
import pickle
scaler=pickle.load(open('std_scaler.pkl','rb'))

def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = scaler.transform(np.array(train, dtype=float))
    y = instrument_list
    return X, y

test = pd.read_csv('test.csv')
X, y = prepare_data_tensorflow(test)

from tensorflow.keras.models import load_model
model = load_model('model.h5')

test_loss, test_acc = model.evaluate(X, y)
print('test_acc: ', test_acc)