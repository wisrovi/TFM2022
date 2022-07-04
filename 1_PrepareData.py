# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


USE_TRAIN_DATA = False
SAVE_CSV = False
TIME_SPLIT = 2 # Sugerido 2 # valor elegido en prueba y error para tener un equilibrio entre cantidad de data y tamaño de cada vector de entrada
QUANTITY_FILES_TO_USE = None # Use None to use as much as possible

from libraries.Kaggle_audios import Kaggle_audios
from libraries.Data import Data

cortar = Kaggle_audios(config_time=TIME_SPLIT, train=USE_TRAIN_DATA)
data = Data(save_csv=SAVE_CSV)


if __name__ == '__main__':
    all_data, all_label, rate = cortar.read_data(limit=QUANTITY_FILES_TO_USE)  # None
    # print(cortar.BASE_INSTRUMENTS) # para interpretar cada posicion de all_label

    print("TOTAL:")
    print("Input:", all_data.shape, " - Output:", all_label.shape, " - rate:", rate)

    # numpy
    data.save_numpy(data=all_data, train=USE_TRAIN_DATA, label=False)
    #all_data = data.read_numpy(train=USE_TRAIN_DATA, label=False)

    data.save_numpy(data=all_label, train=USE_TRAIN_DATA, label=True)
    #all_label = data.read_numpy(train=USE_TRAIN_DATA, label=True)

    # tradicional
    #data.save(data=all_data, train=USE_TRAIN_DATA, label=False)
    #train = data.read(train=USE_TRAIN_DATA, label=False)

    #data.save(data=all_label, train=USE_TRAIN_DATA, label=True, titles=cortar.BASE_INSTRUMENTS)
    #label = data.read(train=USE_TRAIN_DATA, label=True)

    print("GUARDADO COMPLETADO")
    #print("Input:", all_data.shape, " - Output:", all_label.shape)

    print("Rate:", rate)


"""
En data train:
    time_split = 1
        Input: (121554, 44100)  - Output: (121554, 11)
    time_split = 2
        Input: (60851, 88200)  - Output: (60851, 11)
    time_split = 3
        Input: (40625, 132300)  - Output: (40625, 11)
    time_split = 4
        Input: (30508, 176400)  - Output: (30508, 11)
    time_split = 5
        Input: (24440, 220500)  - Output: (24440, 11)        

En data test:
    time_split = 1
        Input: (1485, 44100)  - Output: (1485, 11)
    time_split = 2
        Input: (745, 88200)  - Output: (745, 11)
    time_split = 3
        Input: (499, 132300)  - Output: (499, 11)
    time_split = 4
        Input: (375, 176400)  - Output: (375, 11)
    time_split = 5
        Input: (300, 220500)  - Output: (300, 11)
        
"""