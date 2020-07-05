import pickle
import numpy as np


class Edgelist:
    def __init__(self, graph_id: int):
        self.data = []
        self.graph_id = graph_id
        self.__num_of_rows = 0
        self.__is_pickled = False
        self.max_node = -1

    def load_from_file(self, filepath: str):
        with open(filepath, "r") as file:
            lines = file.readlines()

        self.data = []
        for i in range(len(lines)):
            if lines[i].find(" ") != -1:
                row_data = lines[i].split(" ")
            elif lines[i].find("\t") != -1:
                row_data = lines[i].split("\t")
            else:
                raise ValueError(fr'Unknown separator for edgelist format. The data found is: {lines[i]}')
            v1 = np.int32(row_data[0])
            v2 = np.int32(row_data[1])
            self.add_edge(v1, v2)
            if self.max_node < v1:
                self.max_node = v1
            if self.max_node < v2:
                self.max_node = v2

    def add_edge(self, v1: np.int32, v2: np.int32):
        if self.__is_pickled:
            raise ValueError('Cannot add an edge to already pickled object')

        self.data.append([v1, v2])
        self.__num_of_rows += 1

    def pickle(self, filename):
        print('Serializing data to: ' + filename)
        self.data = np.array(self.data, dtype=np.int32)
        data_dump = [self.data, self.graph_id, self.__num_of_rows]
        with open(filename, 'wb') as f:
            pickle.dump(data_dump, f)
        self.__is_pickled = True

    def unpickle(self, filename):
        print('De-serializing data from: ' + filename)
        with open(filename, 'rb') as f:
            data_dump = pickle.load(f)
            self.data = data_dump[0]
            self.graph_id = data_dump[1]
            self.__num_of_rows = data_dump[2]
