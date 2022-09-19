import scipy.io.wavfile as wav
import pandas as pd


class Corte_audio():
    SEGUNDOS_CORTE = 5

    def __init__(self, config_time: int = 5, save_audios: str = None):
        self.SEGUNDOS_CORTE = config_time
        self.save_audios = save_audios

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

    def ___save_data(self, data_save, file):
        df = pd.DataFrame(data=data_save, columns=['instrument'], index=None, dtype=None, copy=False)
        df.to_csv(file, index=False)

    def split_data(self, path_wav: str = "", path_csv: str = "", tag: str = "dat"):
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

        if self.save_audios is not None:
            for i, data in enumerate(muestras):
                wav.write(self.save_audios + f"{tag}_{i}.wav", rate, data)
                self.___save_data(instrumentos[i], self.save_audios + f"{tag}_{i}.csv")

        return muestras, instrumentos, rate


if __name__ == "__main__":
    corte = Corte_audio(config_time=2, save_audios="data/new_audios/")
    muestras, instrumentos, rate = corte.split_data(path_wav="data/2106.wav", path_csv="data/2106.csv")

    print(instrumentos)
    print(len(muestras), len(instrumentos))
    print(rate)
