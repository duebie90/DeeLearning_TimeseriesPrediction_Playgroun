# Origin: https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/


# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import os

def convert2float(data):
    out_data = []
    try:
        for d in data:
            out_data.append(list(map(float, d)))
        return out_data
    except:
        return None


WEIGHTS_DIR = os.path.join("..","WEIGHTS")
CSV_DIR = os.path.join("..","CSV")
#PREFIX = "simple_MLP_EURUSD_0"
#PREFIX = "simple_MLP_EURUSD_2_epoch"
PREFIX = "LSTM_EURUSD_stateful_sequence_len_10_bs_1_run_0_scaled"

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
# Datum Zeit Eröffnung Hoch Tief Schluss, Ticks
# Row 5 = Schlusskurs
dataframe_train = pandas.read_csv(os.path.join('..','csv_input','EURUSD_train.csv'), usecols=[5], engine='python', skipfooter=3)
dataframe_val = pandas.read_csv(os.path.join('..','csv_input','EURUSD_val.csv'), usecols=[5], engine='python', skipfooter=3)
dataset_train = dataframe_train.values
dataset_val = dataframe_val.values
dataset_train = convert2float(dataset_train)
dataset_val = convert2float(dataset_val)

# split into train and test sets
#train_size = int(len(dataset) * 0.67)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train = np.array(dataset_train)
test = np.array(dataset_val)
print(len(train), len(test))

class LSTMPredictor:
    def __init__(self):
        # reshape into X=t and Y=t+1
        # stateful test
        self.look_back = 1
        self.look_forw = 1

        # window method
        #self.look_back = 10
        #self.look_forw = 1

        self.batch_size = 1
        self.stateful = True
        self.nb_epchs = 10

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(train)
        train_scaled = self.scaler.transform(train)
        test_scaled = self.scaler.transform(test)

        self.trainX, self.trainY = self.create_dataset(train_scaled, self.look_back, self.look_forw)
        self.testX, self.testY = self.create_dataset(test_scaled, self.look_back, self.look_forw)

        self.model_input_shape = (self.batch_size, self.testX.shape[1], self.testX.shape[2])
        self.model = self.create_model()
        # for stateful LSTM batch size 1 makes sense


    def create_dataset(self, dataset, look_back=1, look_forward=1):


        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-look_forward-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back + look_forward, 0])
        #(sample, sequence length, feature)
        dataX = np.reshape(np.array(dataX), (len(dataX), look_back, 1))
        return np.array(dataX), np.array(dataY)

    def create_model(self):
        # design network
        model = Sequential()
        model.add(LSTM(50, batch_input_shape=self.model_input_shape, stateful=self.stateful))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model

    def train(self):
        for i in range(self.nb_epchs if self.stateful else 1):
            print("Training run # " + str(i) + " of " + str(self.nb_epchs))
            self.model.fit(self.trainX, self.trainY, nb_epoch=1 if self.stateful else self.nb_epchs, batch_size=self.batch_size, verbose=1, shuffle=False)
            if self.stateful:
                self.model.reset_states()
        self.model.save_weights(os.path.join(WEIGHTS_DIR, PREFIX + "_final.h5"), overwrite=True)

    def evaluate(self):
        i = 0
        for test_x, test_y in zip(self.testX[:100], self.testY[:100]):
            pred_y = self.model.predict(np.expand_dims(test_x, axis=0))
            print("STEP " + str(i) + "/" + str(len(self.trainX)))
            print("test y: " + str(test_y) + " pred y: " + str(pred_y))
            i += 1

    def initialize_model_for_prediction(self):
        print("Initializing model for inference")
        for i, x in enumerate(self.trainX):
            print(str(i) + "/" + str(len(self.trainX)) +" ->result: " + str(self.scaler.inverse_transform(np.expand_dims(self.model.predict(np.expand_dims(x, axis=0)), axis=0))))
        print("DONE.")

    def predict_multiple_steps(self, test_data=None, timesteps=10):
        """Predicts future timesteps by lopping back prediction results.
        excepcting testdata to be shaped according to the model (one sample)"""
        print("Predicting timesteps from testdata while looping data back...")
        if test_data is not None:
            input_data = test_data
        else:
            input_data = self.testX[100]
        pred_history = [test_data]
        for i in range(timesteps):
            y_pred = self.model.predict(np.expand_dims(input_data, axis=0))
            y_pred_back_scaled = self.scaler.inverse_transform(np.expand_dims(y_pred, axis=0))
            input_data = input_data[1:]
            input_data = np.append(input_data, y_pred, axis=0)
            pred_history.append(y_pred_back_scaled)
        plt.plot(np.array(pred_history))
        print("PRED HISTORY: " + str(pred_history))


if __name__ == "__main__":
    predictor = LSTMPredictor()
    #predictor.model.load_weights(os.path.join(WEIGHTS_DIR, PREFIX + "_final.h5"))
    predictor.train()
    predictor.initialize_model_for_prediction()
    predictor.evaluate()

    predictor.predict_multiple_steps()


# Variants to use the LSTM
#    - Window Method --> time steps = 1
    #- state kept during batch: batch-size = number of timesteps to process
    #  The Keras implementation of LSTMs resets the state of the network after each batch.

    #- stateful: batch size=1
    #- sample, time-steps, features