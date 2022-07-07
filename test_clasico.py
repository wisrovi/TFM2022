import pandas as pd
import numpy as np
import pickle
scaler=pickle.load(open('std_scaler.pkl','rb'))

def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = scaler.transform(np.array(train, dtype=float))
    y = instrument_list
    return X, y


test = pd.read_csv('test.csv')
X_test, y_test = prepare_data_tensorflow(test)


randomforest = pickle.load(open('models_backup/randomforest.pkl', 'rb'))
y_final = randomforest.predict(X_test)


import sklearn.metrics as metrics
print("ACC", metrics.accuracy_score(y_test, y_final))
print("PREC", metrics.precision_score(y_test, y_final, average='micro'))
print("PREC", metrics.precision_score(y_test, y_final, average='macro'))
print("PREC", metrics.precision_score(y_test, y_final, average='weighted'))
print("F1", metrics.f1_score(y_test, y_final, average='micro'))
print("")
print("MSE", metrics.mean_squared_error(y_test, y_final))
print("MAE", metrics.mean_absolute_error(y_test, y_final))
print("R2", metrics.r2_score(y_test, y_final))

resultados = metrics.classification_report(y_test, y_final)
print(resultados)

exit()