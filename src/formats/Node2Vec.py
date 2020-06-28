import pickle
import numpy as np


class Node2Vec:
    def __init__(self):
        self.data: np.ndarray = None

    def load_from_file(self, filepath: str):
        with open(filepath, "r") as file:
            lines = file.readlines()

        for i in range(len(lines)):
            row_data = lines[i].split(" ")
            # Initialize an empty numpy array
            if i == 0:
                row_data = list(map(int, row_data))
                self.data = np.empty([row_data[0], row_data[1]], dtype=np.float)
                continue
            # Read the data and place it in numpy array
            row_data = list(map(np.float, row_data))
            self.data[int(row_data[0]), :] = row_data[1:]

    def pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def unpickle(self, filename):
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)