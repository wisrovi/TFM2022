import json

import pandas as pd
import numpy as np
import base64

from .Constantes.instrumentos import instrumentos
from .ProcessAudio.Features import Features


class Util:

    def __init__(self):
        self.TODOS_LABEL = self.__read_data_label()
        self.y_all = self.decode_prediction([1 for _ in range(len(instrumentos))])

    @staticmethod
    def __read_data_label():
        TODOS_LABEL = pd.read_csv("modelo/all_label.csv", header=0)
        TODOS_LABEL = TODOS_LABEL.to_numpy()
        TODOS_LABEL = TODOS_LABEL.tolist()
        TODOS_LABEL = tuple([d[1] for d in TODOS_LABEL])
        return TODOS_LABEL

    def decode_prediction(self, prediction):
        REAL_PREDICTION = []
        for i, p in enumerate(prediction):
            p = int(p)
            if p == 1:
                usado = self.TODOS_LABEL[i]
                REAL_PREDICTION.append(f"{instrumentos[usado]}({usado})")
        return REAL_PREDICTION

    def preprocess_input(self, file_path):
        processAudio = Features()
        processAudio.set_data(file_path)
        DATA = processAudio.build_all()  # Extrayendo caracteristicas audios, salen 26 caracteristicas
        return DATA

    def audio_to_base64(self, data: list):
        json_encoded_list = json.dumps(data)
        # convertir json_encoded_list al formato para base64
        json_encoded_list = json_encoded_list.encode("utf-8")
        b64_encoded_list = base64.b64encode(json_encoded_list)

        # bytes to string
        b64_encoded_list = b64_encoded_list.decode("utf-8")
        return b64_encoded_list

    def base64_to_audio(self, data: str):
        decoded_list = base64.b64decode(data)
        my_list_again = json.loads(decoded_list)
        return my_list_again
