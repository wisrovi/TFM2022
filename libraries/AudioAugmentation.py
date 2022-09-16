# https://medium.com/@alibugra/audio-data-augmentation-f26d716eee66
# https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
import self
from nlpaug.util import AudioVisualizer

from libraries.ProcessAudio import ProcessAudio

try:
    import librosa
    import librosa.display as librosa_display
except ImportError:
    print("Librosa is not installed. Please install it.")
    import subprocess

    subprocess.call(["pip3", "install", "librosa", "--user", "--upgrade"])
    exit()

import nlpaug.augmenter.audio as naa

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


class AudioAugmentation:
    graph: bool = False

    def __init__(self, audio_file, graph=False, save: str = None):
        self.audio_file = audio_file
        self.graph = graph
        self.save = save
        if isinstance(audio_file, str):
            print("Cargando archivo de audio")
            self.name_file = audio_file.split(".")[-2].split("/")[-1]
            self.data, self.rate = self.__read_audio_file(self.audio_file)
        elif isinstance(audio_file, tuple):
            print("Audio file is a tuple")
            self.data, self.rate, self.name_file = audio_file

    def middleware_in(atributo1: str = "", ):
        def _middleware_in(f):
            def wrapper(self, *args, **kwargs):
                # ejecucion antes de ejecutar la funcion
                print(f"Middleware in {f.__name__} ({atributo1})")

                # ejecucion de la funcion
                ejecucion_ok = True
                try:
                    salida = f(self, *args, **kwargs)
                except Exception as e:
                    ejecucion_ok = False
                    salida = e
                    print(f"Error en la ejecucion de la funcion {f.__name__}: {e}")

                # ejecucion despues de ejecutar la funcion
                if ejecucion_ok:
                    if self.graph:
                        self.plot_audio(salida)

                    if self.save is not None:
                        try:
                            if isinstance(salida, list):
                                print(salida)
                                salida = np.array(salida)
                            self.write_audio_file(self.save + self.name_file + "_" + f"{f.__name__}.wav", salida, self.rate)
                        except Exception as e:
                            print(f"Error al guardar el archivo: {e}")
                            print(type(salida))

                return salida

            return wrapper

        return _middleware_in

    def plot_audio(self, data2=None):
        librosa_display.waveshow(self.data, sr=self.rate, alpha=0.5)
        if data2 is not None:
            librosa_display.waveshow(data2, sr=self.rate, color='r', alpha=0.25)

        plt.tight_layout()
        plt.show()

    @middleware_in(atributo1="loudness_f")
    def loudness(self):
        aug = naa.LoudnessAug()
        augmented_data = aug.augment(self.data)[0]
        return augmented_data

    @middleware_in(atributo1="mask_f")
    def add_mask(self):
        aug = naa.MaskAug(sampling_rate=self.rate, mask_with_noise=False)
        augmented_data = aug.augment(self.data)[0]
        return augmented_data

    @middleware_in(atributo1="pitch_f")
    def pitch(self, fact=(2,3)):
        aug = naa.PitchAug(sampling_rate=self.rate, factor=fact)
        augmented_data = aug.augment(self.data)[0]
        return augmented_data

    @middleware_in(atributo1="original")
    def get_original(self):
        return self.data

    @middleware_in(atributo1="crop")
    def add_crop(self, porcentaje: float = 0.5):
        crop = naa.CropAug(sampling_rate=self.rate)
        augmented_data = crop.augment(self.data)[0]
        return augmented_data

    @staticmethod
    def __read_audio_file(file_path):
        input_length = 16000
        data, rate = librosa.core.load(file_path)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data, rate

    @staticmethod
    def write_audio_file(file, data, sample_rate):
        write(file, sample_rate, data)

    @staticmethod
    def plot_time_series(data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

    @middleware_in(atributo1="ruido")
    def add_noise(self, factor_ruido=0.005):
        noise = np.random.randn(len(self.data))
        data_noise = self.data + factor_ruido * noise
        data_noise = data_noise.astype(type(self.data[0]))  # Cast back to same
        return data_noise

    @middleware_in(atributo1="ruido2")
    def add_noise2(self):
        aug = naa.NoiseAug()
        augmented_data = aug.augment(self.data)[0]
        return augmented_data

    @middleware_in(atributo1="shift")
    def shift(self):
        return np.roll(self.data, 1600)

    @middleware_in(atributo1="stretch")
    def stretch(self, rate_stretch=1.0):
        input_length = 16000
        data = librosa.effects.time_stretch(self.data, rate=rate_stretch)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    @middleware_in(atributo1="speed")
    def speed(self):
        aug = naa.SpeedAug()
        augmented_data = aug.augment(self.data)[0]
        return augmented_data

    @middleware_in(atributo1="normalizador")
    def normalizer(self):
        aug = naa.NormalizeAug(method='minmax')
        augmented_data = aug.augment(self.data)[0]
        return augmented_data

    @middleware_in(atributo1="polarizador")
    def polarizer(self):
        aug = naa.PolarityInverseAug()
        augmented_data = aug.augment(self.data)[0]
        return augmented_data


class Audio_K(AudioAugmentation):

    def __init__(self, file_path, label: list, save: str = "", grafica: bool = False):
        super().__init__(file_path, save=save, graph=grafica)
        self.label = label

    def aumentar(self):
        all_data = [
            self.get_original(),
            self.add_noise(factor_ruido=0.05),
            self.add_noise2(),
            self.stretch(rate_stretch=0.8),
            self.shift(),
            self.add_crop(),
            self.loudness(),
            self.speed(),
            self.normalizer(),
            self.polarizer()
        ]

        all_label = [self.label for _ in range(len(all_data))]
        return all_data, all_label


if __name__ == "__main__":
    data_audio = "data/cat.wav"
    (data, rate) = librosa.core.load(data_audio)
    data_audio = (data, rate, "cat")
    label_audio = [1, 1]

    muestras_wav, instrumentos = Audio_K(data_audio, label=label_audio, save='data/output/').aumentar()

    processAudio = ProcessAudio(rate)
    for id_audio, dat in enumerate(muestras_wav):
        processAudio.set_data(dat)
        caracteristicas = processAudio.get_all(id_audio)  # Extrayendo caracteristicas audios, salen 26 caracteristicas
    print(len(muestras_wav), len(instrumentos))