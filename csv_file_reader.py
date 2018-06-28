import csv
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import random
import numpy as np

class csvReader:
    def __init__(self, csv_path):
        file = open(csv_path, "r")
        self.reader = csv.reader(file, delimiter=',', quotechar='|')

    def read(self):
        output_data = []
        for row in self.reader:
            output_data.append(row)
        return output_data

class CsvDataIterator():
    def __init__(self, filepath, selected_fields=[], selected_rows=[], shuffle=False, data_size=100, batch_size=64):
        # ToDo process header and find out rows from title
        self.selected_rows = selected_rows
        print("Preparing data from csv file")
        self.data = csvReader(filepath).read()
        self.index = 0
        self.shuffle = shuffle
        # size of one data element/ number of lines
        self.data_size =data_size
        # number of data elements in one batch
        self.batch_size = batch_size

    def __next__(self):
        if self.shuffle:
            index = random.randint(len(self.data))
        else:
            if self.index < len(self.data):
                index = self.index
            else:
                raise StopIteration
        # ToDo: Implement batch creation
        out_data = []
        for i in range(0, self.data_size):
            out_data.append([self.data[index + i][row_index] for row_index in self.selected_rows])
        self.index += 1 if not self.shuffle else 0
        return out_data

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
