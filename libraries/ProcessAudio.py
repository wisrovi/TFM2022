import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np


class ProcessAudio(object):
    data = None

    def __init__(self, sr: int = 44100):
        self.mfcc = None
        self.zcr = None
        self.rolloff = None
        self.spec_bw = None
        self.spec_cent = None
        self.rmse = None
        self.chroma_stft = None
        self.sr = sr

    def set_data(self, data):
        self.data = data

    def display_waveform(self):
        if self.data is None:
            return None
        # display waveform
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(self.data, sr=self.sr)

    def get_croma(self):
        if self.data is None:
            return None
        self.chroma_stft = librosa.feature.chroma_stft(y=self.data, sr=self.sr)
        return self.chroma_stft

    def get_rmse(self):
        if self.data is None:
            return None
        self.rmse = librosa.feature.rms(y=self.data)
        return self.rmse

    def get_centroide_espectral(self):
        """centroide espectral"""
        if self.data is None:
            return None
        self.spec_cent = librosa.feature.spectral_centroid(y=self.data, sr=self.sr)
        return self.spec_cent

    def get_ancho_banda_espectral(self):
        if self.data is None:
            return None
        self.spec_bw = librosa.feature.spectral_bandwidth(y=self.data, sr=self.sr)
        return self.spec_bw

    def get_rolloff(self):
        """tambien conocido como reduccion espectral"""
        if self.data is None:
            return None
        self.rolloff = librosa.feature.spectral_rolloff(y=self.data, sr=self.sr)
        return self.rolloff

    def get_cruce_por_cero(self):
        if self.data is None:
            return None
        self.zcr = librosa.feature.zero_crossing_rate(self.data)
        return self.zcr

    def get_mfcc(self):
        if self.data is None:
            return None
        self.mfcc = librosa.feature.mfcc(y=self.data, sr=self.sr)
        return self.mfcc

    def get_all(self, i: int) -> list:
        if self.data is None:
            return []

        self.get_croma()
        self.get_rmse()
        self.get_centroide_espectral()
        self.get_ancho_banda_espectral()
        self.get_rolloff()
        self.get_cruce_por_cero()
        self.get_mfcc()

        data_compresed = f'train{i} {np.mean(self.chroma_stft)} {np.mean(self.rmse)} {np.mean(self.spec_cent)} {np.mean(self.spec_bw)} {np.mean(self.rolloff)} {np.mean(self.zcr)}'
        for e in self.mfcc:
            data_compresed += f' {np.mean(e)}'

        return data_compresed.split()
