import pickle


class Edgelist:
    def __init__(self, graph_id):
        self.data = []
        self.graph_id = graph_id

    def add_edge(self, v1, v2):
        self.data.append([v1, v2, self.graph_id])

    def pickle(self, filename):
        print('Serializing data to: ' + filename)
        data_dump = [self.data, self.graph_id]
        with open(filename, 'wb') as f:
            pickle.dump(data_dump, f)

    def unpickle(self, filename):
        print('De-serializing data from: ' + filename)
        with open(filename, 'rb') as f:
            data_dump = pickle.load(f)
            self.data = data_dump[0]
            self.graph_id = data_dump[1]
