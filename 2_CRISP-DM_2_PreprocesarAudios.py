FOLDER = "data/"

from libraries.Kaggle_audios import Kaggle_audios
from libraries.ProcessAudio import ProcessAudio
from libraries.Decorators import count_elapsed_time
import csv
import pandas as pd


@count_elapsed_time
def extraer_caracteristicas_audios(data, labels, name_file):
    for id_audio, y in enumerate(data):
        x, sr = y, 44100

        processAudio = ProcessAudio()
        processAudio.set_data(x)
        data_save = processAudio.get_all(id_audio)
        data_save += labels[id_audio].tolist()

        with open(FOLDER+name_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_save)


@count_elapsed_time
def convertir_audios_a_numpy(train: bool = True):
    cortar = Kaggle_audios(config_time=TIME_SPLIT, train=train)
    all_data, all_label, rate = cortar.read_data(limit=None)  # None

    # print(cortar.BASE_INSTRUMENTS) # para interpretar cada posicion de all_label

    return all_data, all_label, rate


@count_elapsed_time
def cargando_data_caracteristicas(name_file):
    return pd.read_csv(FOLDER+name_file)


if __name__ == '__main__':
    for TIME_SPLIT in range(13, 31, 1):
        print("\n" * 2)
        print("*"*10, " Time: ", TIME_SPLIT, " ", "*"*10)
        print("\n" * 2)
        """
                        TEST
        """
        print("\n" * 5)
        print("\t", "*" * 50)
        print("\n" * 2)
        print("\t[Test " + str(TIME_SPLIT) + "]:Leyendo audios y convirtiendo a numpy:")
        print("\n" * 2)
        all_data, all_label, rate = convertir_audios_a_numpy(train=False)
        print("\tTOTAL TEST:")
        print("\tTime:", TIME_SPLIT, " - Input:", all_data.shape, " - Output:", all_label.shape, " - rate:", rate)

        print("\n" * 5)
        print("\t", "*" * 50)
        print("\n" * 2)
        print("\t[Test " + str(TIME_SPLIT) + "]:Extrayendo caracteristicas:")
        print("\n" * 2)
        NAME_FILE_TEST = 'test_' + str(TIME_SPLIT) + 'seg' + '.csv'
        extraer_caracteristicas_audios(all_data, all_label, NAME_FILE_TEST)




        """
                        TRAIN
        """
        print("\n" * 5)
        print("\t", "*" * 50)
        print("\n" * 2)
        print("\t[Train " + str(TIME_SPLIT) + "]:Leyendo audios y convirtiendo a numpy:")
        print("\n" * 2)
        all_data, all_label, rate = convertir_audios_a_numpy(train=True)
        print("\tTOTAL TRAIN:" + str(TIME_SPLIT))
        print("\tTime:", TIME_SPLIT, " - Input:", all_data.shape, " - Output:", all_label.shape, " - rate:", rate)

        print("\n" * 5)
        print("\t", "*" * 50)
        print("\n" * 2)
        print("\t[Train " + str(TIME_SPLIT) + "]:Extrayendo caracteristicas:")
        print("\n" * 2)
        NAME_FILE_TRAIN = 'train_' + str(TIME_SPLIT) + 'seg' + '.csv'
        extraer_caracteristicas_audios(all_data, all_label, NAME_FILE_TRAIN)


        """
                        DATA CARACTERISTICAS
        """
        print("\n" * 5)
        print("\t", "*" * 50)
        print("\n" * 2)
        print("\t[Train " + str(TIME_SPLIT) + "]:Cargando data caracteristicas:")
        print("\n" * 2)
        train = cargando_data_caracteristicas(NAME_FILE_TRAIN)
        print("Time:", TIME_SPLIT, " - train:", train.shape, " - Rate:", rate)

        print("\n" * 5)
        print("\t", "*" * 50)
        print("\n" * 2)
        print("\t[Test " + str(TIME_SPLIT) + "]:Cargando data caracteristicas:")
        print("\n" * 2)
        test = cargando_data_caracteristicas(NAME_FILE_TEST)
        print("\t[Time:", TIME_SPLIT, " - train:", test.shape, " - Rate:", rate)

"""
En data train:
    time_split = 1
        
    time_split = 2
        
    time_split = 3
        
    time_split = 4
        
    time_split = 5
           

En data test:
    time_split = 1
        
    time_split = 2
        
    time_split = 3
        
    time_split = 4
        
    time_split = 5
        
        
"""
