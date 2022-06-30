# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import scipy.io.wavfile as wav
import pandas as pd


class Corte_audio():
    SEGUNDOS_CORTE = 5

    def __init__(self, config_time:int=5):
        self.SEGUNDOS_CORTE = config_time

    def __hallar_instrumentos(self, corte_start, corte_end, instrument, start_time):
        acumulador = list()
        for (i, time_start) in enumerate(start_time):
            if corte_end >= start_time[i] >= corte_start:
                # print(start_time[i], end_time[i], instrument[i])
                if not instrument[i] in acumulador:
                    acumulador.append(instrument[i])
            else:
                if time_start > corte_end:
                    break
        return acumulador

    def split_data(self, path_wav:str= "", path_csv:str= ""):
        (rate, sig) = wav.read(path_wav)
        labels = pd.read_csv(path_csv)

        instrument = labels['instrument']
        start_time = labels['start_time']

        pivote = 0
        corte = rate * self.SEGUNDOS_CORTE

        muestras = list()
        instrumentos = list()
        for _ in range(round(len(sig) / corte)):
            corte_start, corte_end = pivote, pivote + corte
            data = sig[corte_start:corte_end]
            muestras.append(data)
            instrumentos.append(self.__hallar_instrumentos(corte_start, corte_end, instrument, start_time))
            pivote += corte

        if pivote < len(sig):
            data = sig[pivote:]
            muestras.append(data)
            instrumentos.append(self.__hallar_instrumentos(pivote, len(sig), instrument, start_time))

        return muestras, instrumentos

import os
class Kaggle_audios(Corte_audio):
    BASE_FOLDER = "/kaggle/input/musicnet-dataset/musicnet/musicnet/"
    path_wav = "_data/"
    path_csv = "_labels/"

    def __init__(self, config_time:int=5, train: bool = True):
        super().__init__(config_time=config_time)
        group_to_use = "test"
        if train:
            group_to_use = "train"

        base_path = os.getcwd().split("musicnet")[0]+"musicnet"
        self.path_wav = base_path + self.BASE_FOLDER + group_to_use + self.path_wav
        self.path_csv = base_path + self.BASE_FOLDER + group_to_use + self.path_csv

        self.file_wav = os.listdir(self.path_wav)
        # print(len(self.file_wav))

    def leer_path_data_label(self, id:int=0):
        audio_usar = self.path_wav + self.file_wav[id]
        resource_read = audio_usar.split(".")[0].split("/")[-1]
        label_usar = self.path_csv + resource_read + ".csv"
        return audio_usar, label_usar, resource_read

    def leer_data_label(self, id:int=0):
        audio_usar, label_usar, resource_read = self.leer_path_data_label(id)

        muestras_wav, instrumentos = self.split_data(audio_usar, label_usar)
        return (muestras_wav, instrumentos, resource_read)

    def read_data(self, limit=None):
        all_data = list()
        all_label = list()

        partial = len(self.file_wav)
        if limit is not None:
            if limit <= 0:
                limit = 1
            if limit > partial:
                limit = partial
            partial = len(self.file_wav[:limit])

        for i in range(partial):
            print("*"*10)
            muestras_wav, instrumentos, resource_read = self.leer_data_label(i)
            print("Reading: id:", i, " - file:", resource_read)
            #print(resource_read + ".wav", len(muestras_wav), type(muestras_wav))
            #print(resource_read + ".csv", len(instrumentos), type(instrumentos))
            all_data += muestras_wav
            all_label += instrumentos

            #audio_usar, label_usar, resource_read = self.leer_path_data_label(i)
            #print(audio_usar)


        return all_data, all_label


cortar = Kaggle_audios(config_time=5, train=True)


import csv
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    all_data, all_label = cortar.read_data(limit=2) # None
    print(len(all_data), len(all_label))
    print(len(all_data[0]), len(all_label[0]))
    print(all_data[0][0], all_label[0])

    with open('data.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(all_data)

    with open('label.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(all_label)


if __name__ == '__main__':
    print_hi('PyCharm')


