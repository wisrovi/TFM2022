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

        return muestras, instrumentos, rate