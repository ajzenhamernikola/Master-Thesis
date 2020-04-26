import pickle
import numpy as np


class Edgelist:
    def __init__(self, graph_id: int):
        self.data = []
        self.graph_id = graph_id
        self.__num_of_rows = 0
        self.__is_picked = False

    def add_edge(self, v1, v2):
        if self.__is_picked:
            raise ValueError('Cannot add an edge to already pickled object')

        self.data.append([v1, v2, self.graph_id])
        self.__num_of_rows += 1

    def pickle(self, filename):
        print('Serializing data to: ' + filename)
        self.data = np.array(self.data, dtype=np.int32)
        data_dump = [self.data, self.graph_id, self.__num_of_rows]
        with open(filename, 'wb') as f:
            pickle.dump(data_dump, f)
        self.__is_picked = True

    def unpickle(self, filename):
        print('De-serializing data from: ' + filename)
        with open(filename, 'rb') as f:
            data_dump = pickle.load(f)
            self.data = data_dump[0]
            self.graph_id = data_dump[1]
            self.__num_of_rows = data_dump[2]
