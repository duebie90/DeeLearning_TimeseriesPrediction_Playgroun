import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge, Activation, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, Reshape, Deconvolution2D
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

from csv_file_reader import csvReader
import os

class KerasApplication:
    def __init__(self):
        self.cv2_file = os.path.join("csv_input", "EURUSD1.csv")
        self.csv_reader = csvReader
    def model_descr(self):
        input_layer = Input()
        x = input_layer
        Model(input_layer, x)
    def get_data_iterator(self):
        pass
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



