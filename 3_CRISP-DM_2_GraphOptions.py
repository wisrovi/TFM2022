import pandas as pd
import matplotlib.pyplot as plt
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

result_test = dict()
for file in files:
    if file.find("test")>=0:
        test = pd.read_csv(base_path+file)
        X, y = prepare_data_tensorflow(test)
        result_test[int(file.split("_")[1].split("seg")[0])] = {
            "file":file,
            "shape": (X.shape, y.shape)
        }

result_test = dict(sorted(result_test.items()))

result_train = dict()
for file in files:
    if file.find("train")>=0:
        test = pd.read_csv(base_path+file)
        X, y = prepare_data_tensorflow(test)
        result_train[int(file.split("_")[1].split("seg")[0])] = {
            "file":file,
            "shape": (X.shape, y.shape)
        }

result_train = dict(sorted(result_train.items()))



"""
        Iniciando a graficar
"""

x_train_np = list()
for k, v in result_train.items():
    x_train_np.append(v.get('shape')[0][0]+v.get('shape')[1][0])
y_train_np = [i+1 for i in range(len(x_train_np))]
y_train_np = np.array(y_train_np)
x_train_np = np.array(x_train_np)


x_test_np = list()
for k, v in result_test.items():
    x_test_np.append(v.get('shape')[0][0]+v.get('shape')[1][0])
y_test_np = [i+1 for i in range(len(x_test_np))]
y_test_np = np.array(y_test_np)
x_test_np = np.array(x_test_np)

base_y = [0 for _ in range(len(x_test_np))]

fig = plt.figure()
subplot1=fig.add_subplot(2,1,1)
subplot2=fig.add_subplot(2,1,2)

subplot1.plot(y_test_np, x_test_np, label="test")
subplot1.plot(y_test_np, base_y, "-")
subplot1.fill_between(y_test_np, y_train_np, x_test_np, color='green', alpha=0.5)
subplot1.grid()
subplot1.legend()

ids = [str(i) for i in x_train_np]
subplot2.plot(y_train_np, x_train_np, label="train")
subplot2.plot(y_test_np, base_y, "-")
subplot2.fill_between(y_test_np, y_train_np, x_train_np, color='green', alpha=0.5)
subplot2.grid()
subplot2.legend()

plt.ylabel("Cantidad de datos")
plt.xlabel("segundos split")
fig.suptitle("Cantidad de datos segun el split elegido")


plt.show()
fig.savefig("cantidad_datos_segun_split_segundos.jpg")

data = list()
for i, v in enumerate(y_train_np):
    data_save = [ y_train_np[i], x_train_np[i], x_test_np[i] ]
    data.append(data_save)

df = pd.DataFrame(data, columns=['Division_segundos', 'Cantidad_datos_train', 'Cantidad_datos_test'])

df.to_csv("cantidad_datos_segun_split_segundos.csv")
















data = [
    [1, 44100],
    [2, 88200],
    [3, 132300] ,
    [4, 176400],
    [5, 220500]
]

x1, x2 = data[0][0], data[-1][0]
y1, y2 = data[0][1], data[-1][1]
m = (y1-y2)/(x1-x2)
b = (y2) - m *x2

for x in range(6, 13, 1):
    y = int(m*x + b)
    #print(x, y)
    data.append([x, y])



df = pd.DataFrame(data, columns=['Division_segundos', 'Cantidad_entradas'])
print(df)
df.to_csv("cantidad_entradas_segun_split_segundos.csv")



from matplotlib import pyplot as plt
fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot([i[0] for i in data], [i[1] for i in data])
ax.plot(y_test_np, base_y, "-")
ax.fill_between(y_test_np, y_train_np, [i[1] for i in data], color='green', alpha=0.5)
plt.ylabel("Cantidad de entradas")
plt.xlabel("segundos split")
plt.show()
fig.savefig("cantidad_entradas_segun_split_segundos.jpg")



