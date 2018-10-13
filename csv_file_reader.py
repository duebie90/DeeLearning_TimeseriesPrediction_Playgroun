import csv
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import random
import numpy as np
from keras.callbacks import Callback

class csvReader:
    def __init__(self, csv_path):
        file = open(csv_path, "r")
        self.reader = csv.reader(file, delimiter=',', quotechar='|')

    def read(self):
        output_data = []
        for row in self.reader:
            output_data.append(row)
        return output_data

class CsvDataIterator(Callback):
    def __init__(self, filepath, selected_fields=[], selected_rows=[], shuffle=False, data_size=100, batch_size=64):
        # ToDo process header and find out rows from title
        self.selected_rows = selected_rows
        print("Preparing data from csv file")
        self.data = csvReader(filepath).read()
        # number of data elements in one batch
        self.batch_size = batch_size
        self.nbatches = np.floor(len(self.data) / self.batch_size)-1
        self.index = 0
        self.shuffle = shuffle
        # size of one data element/ number of lines
        self.data_size =data_size



    def convert2float(self, data):
        out_data = []
        try:
            for d in data:
                out_data.append(list(map(float, d)))
            return out_data
        except:
            return None


    def __next__(self):
        batch_x = []
        batch_y = []
        while(len(batch_x) < self.batch_size):
            if self.shuffle:
                index = random.randint(len(self.data))
            else:
                if self.index < len(self.data):
                    index = self.index
                else:
                    raise StopIteration
            if len(self.data) >= index + 100:
                pass
            else:
                print("data end")
            out_data_x = []
            for i in range(0, self.data_size):
                out_data_x.append([self.data[index + i][row_index] for row_index in self.selected_rows])

            out_data_x = self.convert2float(out_data_x)

            #out_data_x = np.asarray(np.expand_dims(out_data_x, axis=0), dtype=np.float32)
            out_data_x = np.asarray(out_data_x, dtype=np.float32)

            # the newest value is used as label
            y = np.expand_dims([self.data[index + i + 1][row_index] for row_index in self.selected_rows], axis=0)

            y = self.convert2float(y)
            y = np.asarray(y, dtype=np.float32)

            # TEST
            y = np.expand_dims(np.asarray(np.mean(y)), axis=0)

            # normalize x to 0..1
            min_x = np.min(out_data_x)
            out_data_x -= min_x
            y -= min_x

            max_x = np.max(out_data_x)
            out_data_x /= max_x
            y /= max_x

            self.index += 1 if not self.shuffle else 0
            #print("nect(): y = " + str(y[0]))
            batch_x.append(out_data_x)

            # use difference from last entry from history as y_true
            y_true = y - np.mean(out_data_x[-1])

            batch_y.append(y_true)




        return np.asarray(batch_x), np.asarray(batch_y)

    def __iter__(self):
        return self





class EventHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_modified(self, event):
        self.callback()


class FileObserver:
    def __init__(self, path, callback):
        self.path = path
        self.callback = callback
        self.event_handler = EventHandler(callback)
        self.observer = Observer()
        self.observe()

    def observe(self):
        self.observer.schedule(self.event_handler, self.path, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()



def callback():
    reader = csvReader(r"csv_input/EURUSD1.csv")
    data = reader.read()
    print(str(data))



if __name__ == "__main__":
    #reader = csvReader("EURUSD1.csv")
    #data = reader.read()

    fo = FileObserver(r"C:\Users\admin\AppData\Roaming\MetaQuotes\Terminal\76AE827A66F7801B9D79B1FD1D2103FD\MQL4\Files", callback)
