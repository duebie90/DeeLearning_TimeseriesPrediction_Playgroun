import numpy as np
from sklearn.preprocessing import MinMaxScaler

def convert2float(data):
    out_data = []
    try:
        for d in data:
            out_data.append(list(map(float, d)))
        return out_data
    except:
        return None


def create_dataset(self, dataset, look_back=1, look_forward=1, sequence_length=1):
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back + look_forward, 0])