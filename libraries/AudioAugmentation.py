# https://medium.com/@alibugra/audio-data-augmentation-f26d716eee66
# https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
import self

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
from functools import wraps


class AudioAugmentation:
    graph: bool = False

    def __init__(self, audio_file, graph=False, save: str = None):
        self.name_file = audio_file.split(".")[-2].split("/")[-1]
        self.audio_file = audio_file
        self.graph = graph
        self.save = save
        self.data, self.rate = self.__read_audio_file(self.audio_file)

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
                if self.graph:
                    self.plot_audio(salida)

                if self.save is not None:
                    self.write_audio_file(self.save + self.name_file + "_" + f"{f.__name__}.wav", salida, self.rate)

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
        augmented_data = aug.augment(self.data)
        return augmented_data

    @middleware_in(atributo1="original")
    def get_original(self):
        return self.data

    def crop(self):
        crop = naa.CropAug(sampling_rate=self.rate, max_percentage=0.5)
        augmented_data = crop.augment(self.data)
        if self.graph:
            self.plot(augmented_data)
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

    @middleware_in(atributo1="shift")
    def shift(self):
        return np.roll(self.data, 1600)

    def manipulate(self, data, shift_max, shift_direction='right'):
        shift = np.random.randint(self.rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    @middleware_in(atributo1="stretch")
    def stretch(self, rate_stretch=1.0):
        input_length = 16000
        data = librosa.effects.time_stretch(self.data, rate=rate_stretch)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data


if __name__ == "__main__":
    # Create a new instance from AudioAugmentation class
    aa = AudioAugmentation("data/cat.wav", graph=False, save='data/output/')
    original = aa.get_original()  # Get original audio
    data_noise = aa.add_noise(factor_ruido=0.05)  # Adding noise to sound
    data_stretch = aa.stretch(rate_stretch=0.8)  # Stretching the sound
    data_roll = aa.shift()  # Shifting the sound

    exit()