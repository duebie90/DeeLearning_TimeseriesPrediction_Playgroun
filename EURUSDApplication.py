import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge, Activation, UpSampling2D
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling2D, Dropout, Reshape, Deconvolution2D
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from csv_file_reader import csvReader, CsvDataIterator
import os
import KerasApplication
import matplotlib.pyplot as plt

WEIGHTS_DIR = "WEIGHTS"
PREFIX = "differential_y_pred_exp0"

class EURUSDApplication:
    def __init__(self):
        #super(EURUSDApplication, self).__init__()
        self.csv_file_train = os.path.join("csv_input", "EURUSD_train.csv")
        self.csv_file_validation = os.path.join("csv_input", "EURUSD_val.csv")
        self.csv_file_test = os.path.join("csv_input", "EURUSD1.csv")
        self.batch_size = 64
        self.input_length = 100
        self.nb_epoch = 10
        self.model = None

    def get_model_descr(self, filters=[32,32,32,32]):
        input_layer = Input(shape=(self.input_length,4))
        x = input_layer
        for f_count in filters:
            x = Convolution1D(f_count, 5, activation="relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        #output = Dense(4, activation="linear")(x)
        output = Dense(1, activation="linear")(x)
        model = Model(input_layer, output)
        model.summary()
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

    def load_model(self, filename):
        self.model = self.get_model_descr()
        self.model.load_weights(os.path.join(WEIGHTS_DIR, filename))

    def get_training_iterators(self):
        train = CsvDataIterator(self.csv_file_train, selected_rows=[2,3,4,5], batch_size=self.batch_size, data_size=self.input_length)
        val = CsvDataIterator(self.csv_file_validation, selected_rows=[2, 3, 4, 5], batch_size=self.batch_size, data_size=self.input_length)
        return [train, val]

    def train(self):
        checkpoint = ModelCheckpoint(filepath=os.path.join(WEIGHTS_DIR, PREFIX + "_{epoch:04d}.hdf5"), save_best_only=True,
                                save_weights_only=True)
        csv_logger = CSVLogger(filename='results_' + PREFIX + '.csv', append=True)

        es = EarlyStopping(patience=10)

        model = self.get_model_descr()
        train, val = self.get_training_iterators()
        model.fit_generator(train,
                            samples_per_epoch=64,#train.nbatches * self.batch_size,
                            #samples_per_epoch=10,
                            nb_epoch=self.nb_epoch,
                            initial_epoch=0,  # (epoch index to start with[0..n])
                            nb_worker=1,
                            max_q_size=1,
                            pickle_safe=False,
                            callbacks=[train, checkpoint, csv_logger, es],
                            verbose=1,
                            nb_val_samples=val.nbatches * self.batch_size,
                            #nb_val_samples=200,
                            validation_data=val)
        model.save_weights(os.path.join(WEIGHTS_DIR, PREFIX + "_final.h5"), overwrite=True)


    def write_test_data(self, dataset="test", nb_batches=10):
        pass

    def check_training_data(self, dataset="test", nb_batches=10, do_inference=False):
        if dataset == "train":
            iterator, _ = app.get_training_iterators()
        elif dataset == "validation":
            _, iterator = app.get_training_iterators()
        elif dataset == "test":
            raise NotImplementedError

        for data in iterator:
            x = data[0][0]
            y_true = data[1][0]
            last_x_mean = np.mean(x[-1])

            plt.figure("Sequence " + dataset)
            if do_inference and self.model is not None:
                y_pred = self.model.predict(np.expand_dims(x, axis=0))[0]

                # resolve y_pred according to difference
                y_pred_plot_value = y_pred + last_x_mean

                plt.plot([100], [y_pred_plot_value], 'go')

            for i in range(x.shape[1]):
                # x values
                plt.plot(x[:,i])
                # y_value
                #plt.plot([10], [data[1][0]], 'ro')
            # resolve y_true according to difference
            y_true_plot_value = y_true + last_x_mean

            plt.plot([100], [y_true_plot_value], 'ro')

            plt.show()


    def load_weights(self):
        pass
    def predict(self, data):
        pass



if __name__ == "__main__":
    app = EURUSDApplication()
    #app.load_model("exp0_final.h5")
    #app.check_training_data(dataset="train", do_inference=True)
    app.train()

    quit()
    iterator, _ = app.get_training_iterators()
    for data in iterator:
        print(str(data))
