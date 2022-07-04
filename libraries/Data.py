

import pickle
import csv

# https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
from numpy import savez_compressed
from numpy import load


class Data(object):
    def __init__(self, save_csv:bool = False):
        self.save_csv = save_csv

    def save_numpy(self, data, train: bool = True, label: bool = False, titles:list = None):
        name = "train"
        if not train:
            name = "test"

        name_file = 'data_' + name
        if label:
            name_file = 'data_' + name + '_label'

        savez_compressed("data_partial/" + name_file + '.npz', data)

        if self.save_csv:
            with open('data_partial/' + name_file + '.csv', 'w') as f:
                write = csv.writer(f)
                if titles is not None:
                    write.writerow(titles)
                for i in range(len(data)):
                    write.writerow(data[i])

    def read_numpy(self, train: bool = True, label: bool = False):
        name = "train"
        if not train:
            name = "test"

        name_file = 'data_' + name
        if label:
            name_file = 'data_' + name + '_label'

        dict_data = load("data_partial/" + name_file + '.npz')
        data = dict_data['arr_0']
        return data

    def save(self, data, train: bool = True, label: bool = False, titles:list = None):
        name = "train"
        if not train:
            name = "test"

        name_file = 'data_' + name
        if label:
            name_file = 'data_' + name + '_label'

        with open("data_partial/" + name_file + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    def read(self, train: bool = True, label: bool = False):
        name = "train"
        if not train:
            name = "test"

        name_file = 'data_' + name
        if label:
            name_file = 'data_' + name + '_label'

        data = None
        with open("data_partial/" + name_file + '.pkl', 'rb') as f:
            data = pickle.load(f)

        return data