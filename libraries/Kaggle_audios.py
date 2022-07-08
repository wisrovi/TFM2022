from libraries.Corte_audio import Corte_audio

import os
import numpy as np


class Kaggle_audios(Corte_audio):
    BASE_FOLDER = "/kaggle/input/musicnet-dataset/musicnet/musicnet/"
    path_wav = "_data/"
    path_csv = "_labels/"

    BASE_INSTRUMENTS = [1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74]

    def __init__(self, config_time: int = 5, train: bool = True):
        super().__init__(config_time=config_time)
        group_to_use = "test"
        if train:
            group_to_use = "train"

        base_path = os.getcwd().split("musicnet")[0] + "musicnet"
        base_path = ""  # usar para cuando se trabaja desde jupyter
        self.path_wav = base_path + self.BASE_FOLDER + group_to_use + self.path_wav
        self.path_csv = base_path + self.BASE_FOLDER + group_to_use + self.path_csv

        print(self.path_wav)
        print(self.path_csv )
        self.file_wav = os.listdir(self.path_wav)

    def leer_path_data_label(self, id: int = 0):
        audio_usar = self.path_wav + self.file_wav[id]
        resource_read = audio_usar.split(".")[0].split("/")[-1]
        label_usar = self.path_csv + resource_read + ".csv"
        return audio_usar, label_usar, resource_read

    def leer_data_label(self, id: int = 0):
        audio_usar, label_usar, resource_read = self.leer_path_data_label(id)

        muestras_wav, instrumentos, rate = self.split_data(audio_usar, label_usar)
        return (muestras_wav, instrumentos, rate, resource_read)

    def __correcion_instrumentos(self, instrumentos):
        for id, usado_list in enumerate(instrumentos):
            instrumentos_usados = [0 for _ in self.BASE_INSTRUMENTS]
            for usado in usado_list:
                i = self.BASE_INSTRUMENTS.index(usado)
                instrumentos_usados[i] = 1
            instrumentos[id] = instrumentos_usados
        return instrumentos

    def __correccion_data(self, muestras_wav):
        c = [0 for _ in range(len(muestras_wav[-2]) - len(muestras_wav[-1]))]
        muestras_wav[-1] = np.array(muestras_wav[-1].tolist() + c)
        return muestras_wav

    def read_data(self, limit=None, show_info:bool = True):
        all_data = list()
        all_label = list()
        rate = None

        partial = len(self.file_wav)
        if limit is not None:
            if limit <= 0:
                limit = 1
            if limit > partial:
                limit = partial
            partial = len(self.file_wav[:limit])

        if show_info:
            print("\t\t", "*" * 20, end="")
        for i in range(partial):
            if show_info:
                print("\t\t", "*"  * 10)
            muestras_wav, instrumentos, rate, resource_read = self.leer_data_label(i)
            if show_info:
                print("\t\tReading: id:", f"{i}/{partial}", " - file:", resource_read, " - len before:", len(all_data))
            # print(resource_read + ".wav", len(muestras_wav), type(muestras_wav))
            # print(resource_read + ".csv", len(instrumentos), type(instrumentos))

            all_data += self.__correccion_data(muestras_wav)
            all_label += self.__correcion_instrumentos(instrumentos)

            try:
                if i % 100 == 0:
                    print("\n\t", end="")
                print(".", end=" ")
            except:
                print()

        all_label = np.array(all_label)
        all_data = np.array(all_data)

        if show_info:
            print("*"*30)

        return all_data, all_label, rate
