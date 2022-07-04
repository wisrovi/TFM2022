USE_TRAIN_DATA = True

from libraries.Data import Data

data = Data()

if __name__ == '__main__':
    print("Cargando datos train")
    all_data = data.read_numpy(train=USE_TRAIN_DATA, label=False)
    print("Cargando datos train label")
    all_label = data.read_numpy(train=USE_TRAIN_DATA, label=True)

    print("GUARDADO COMPLETADO")
    print("Input:", all_data.shape, " - Output:", all_label.shape)

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