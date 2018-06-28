import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge, Activation, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, Reshape, Deconvolution2D
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

from csv_file_reader import csvReader, CsvDataIterator
import os
import KerasApplication
import matplotlib.pyplot as plt

class EURUSDApplication:
    def __init__(self):
        #super(EURUSDApplication, self).__init__()
        self.csv_file_train = os.path.join("csv_input", "EURUSD1.csv")
        self.csv_file_validation = os.path.join("csv_input", "EURUSD1.csv")
        self.csv_file_test = os.path.join("csv_input", "EURUSD1.csv")
        self.batch_size = 1
        self.input_length = 100

    def model_descr(self):
        input_layer = Input(shape=(200,4))
        x = input_layer
        Model(input_layer, x)

    def get_training_iterators(self):
        train = CsvDataIterator(self.csv_file_train, selected_rows=[2,3,4,5], batch_size=self.batch_size, data_size=self.input_length)
        val = CsvDataIterator(self.csv_file_validation, selected_rows=[2, 3, 4, 5], batch_size=self.batch_size, data_size=self.input_length)
        return [train, val]

    def train(self):
        checkpoint = ModelCheckpoint()
        pass
    def write_test_data(self, dataset="test", nb_batches=10):
        pass
    def check_training_data(self, dataset="test", nb_batches=10):

        pass
    def load_weights(self):
        pass
    def predict(self, data):
        pass



if __name__ == "__main__":
    app = EURUSDApplication()
    iterator, _ = app.get_training_iterators()
    for data in iterator:
        print(str(data))
