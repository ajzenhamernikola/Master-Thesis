import os
import random
import math

import numpy as np
import pandas as pd
from dgl import DGLGraph
from dgl.data import save_graphs, load_graphs
from graphvite import graph as vite_graph
from graphvite import solver as vite_solver
from torch.utils.data import Dataset

from src.formats.Edgelist import Edgelist
from src.formats.SparseMatrix import SparseMatrix


def log10_transform_data(data):
    minimum_log10_value = 0.001
    data[data < minimum_log10_value] = minimum_log10_value
    return np.log10(data)


class CNFDatasetNode2Vec(Dataset):
    def __init__(self, csv_file_x: str, csv_file_y: str, root_dir: str, data_type: str = "train", train: float = 0.8,
                 val: float = 0.1, test: float = 0.1):
        super(CNFDatasetNode2Vec).__init__()
        # Checks
        percents = [train, val, test]
        mapped = list(map(lambda x: 0 < x <= 1.0, percents))
        filtered = list(filter(lambda x: x is False, mapped))
        valid_percents = len(filtered) == 0
        if not valid_percents:
            raise ValueError(f"All percents must be in range (0.0, 1.0]. You passed: {percents}")
        total_percent = sum(percents)
        if total_percent > 1.0:
            raise ValueError(f"Sum of all percents must be <= 1.0. You passed: {percents} whose sum is {total_percent}")
        valid_data_types = ["train", "val", "test", "train+val"]
        if data_type not in valid_data_types:
            raise ValueError(f"Argument data_type must be one of {valid_data_types}. You passed: {data_type}")

        # Init
        self.root_dir = root_dir
        self.ys = []
        self.csv_data_x = pd.read_csv(csv_file_x)
        self.csv_data_y = pd.read_csv(csv_file_y)
        self.data_dim = 32
        self.indices = []
        self.data_dir = "data"
        self.dataset_type = "node2vec"
        # Keep track of unsuccessfully loaded data, so we can skip them faster
        self.__unsuccessful_txt = os.path.join(os.path.dirname(__file__), "..", "..", self.data_dir,
                                               f"{self.dataset_type}_unsuccessful.txt")
        self.__load_already_known_unsuccessful_graphs()
        print(f"\nPreparing the dataset for phase: {data_type}")

        # Create the folder for pickling data
        csv_x = csv_file_x[csv_file_x.rfind(os.sep, 0, -1) + 1:-4]
        assert (csv_x != "" and csv_x.find(os.sep) == -1)
        csv_x_folder = os.path.join(os.path.dirname(__file__), "..", "..", self.data_dir, csv_x)
        if not os.path.exists(csv_x_folder):
            os.makedirs(csv_x_folder)
        self.csv_x_folder = csv_x_folder

        # Load the data
        n = len(self.csv_data_x)
        indices = list(range(n))
        random.seed(0)
        random.shuffle(indices)
        if data_type == "train":
            low = 0
            high = int(math.floor(train * n))
        elif data_type == "val":
            low = int(math.floor(train * n))
            high = int(math.floor((train + val) * n))
        elif data_type == "train+val":
            low = 0
            high = int(math.floor((train + val) * n))
        else:
            low = int(math.floor((train + val) * n))
            high = n
        indices = indices[low:high]

        # Pickle the graphs if they don't exist
        # print('\nPickling the graph data that doesn\'t exist...')
        for i in indices:
            # print(f"(Graph) Checking the pickled state of instance num {i}...")
            if self.check_if_pickled(i):
                # print(f"\tAlready pickled!")
                continue

            # Pickle graph data
            self.create_edgelist_from_instance_id(i)

        # Pickle the features if they don't exist
        # print('\nPickling the feature data that doesn\'t exist...')
        for i in indices:
            # print(f"(Features) Checking the pickled state of instance num {i}...")
            # If the data is pickled, then we have everything we need, so save the index
            if self.check_if_pickled_features(i):
                # print(f"\tAlready pickled!")
                self.indices.append(i)
                continue

            # If the graph is known to be unsuccessful, skip it
            if self.__is_unsuccessful_graph(i):
                # print(f"\tIs known to be unsuccessful... Skipping this instance!")
                continue

            # Finally, try to pickle feature data and save the index only if we succeed
            save_indices = self.create_node2vec_features(i)
            if save_indices:
                self.indices.append(i)
            else:
                self.__commit_new_unsuccessful_graph(i)

        # Load ys
        final_indices = []
        for i in self.indices:
            # Get the cnf file path
            instance_id: str = self.csv_data_x['instance_id'][i]
            ys = self.csv_data_y[self.csv_data_y['instance_id'] == instance_id]
            ys = np.array(ys.drop(columns=['instance_id']).iloc[0], dtype=np.float32)
            non_solvable = np.all(ys == 1200.0)
            if non_solvable:
                continue
            ys = log10_transform_data(ys)
            self.ys.append(ys)
            final_indices.append(i)

        self.indices = final_indices

    def __commit_new_unsuccessful_graph(self, i):
        self.__unsuccessful_indices.append(i)
        with open(self.__unsuccessful_txt, 'w') as f:
            f.write(" ".join(map(lambda num: str(num), self.__unsuccessful_indices)))

    def __load_already_known_unsuccessful_graphs(self):
        if not os.path.exists(self.__unsuccessful_txt):
            self.__unsuccessful_indices = []
            return

        with open(self.__unsuccessful_txt, 'r') as f:
            self.__unsuccessful_indices = list(map(lambda numstr: int(numstr), f.read().split(" ")))

    def __is_unsuccessful_graph(self, i):
        return i in self.__unsuccessful_indices

    def load_pickled_graph(self, i):
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i)
        g, _ = load_graphs(pickled_filename)
        g = g[0]
        return g

    def load_pickled_features(self, i):
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i, features=True)
        return np.load(pickled_filename + '.npy')

    def create_node2vec_features(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        edgelist_filename = os.path.join(self.root_dir, instance_id + '.edgelist')

        # Prepare graph for Node2Vec
        # print(f"\tPreparing graph for Node2vec...")
        v_graph = vite_graph.Graph()
        v_graph.load(edgelist_filename, as_undirected=False)
        v_graph_node_num = len(v_graph.id2name)

        # Prepare graph for DGL
        # print(f"\tPreparing graph for DGL...")
        g = self.load_pickled_graph(i)
        g_node_num = g.number_of_nodes()

        # Check if graphs have the same number of nodes
        if v_graph_node_num != g_node_num:
            # print(f"\tMismatching number of nodes: {v_graph_node_num} in graphvite != {g_node_num} in dgl")
            return False

        # print(f"\tNumber of nodes: {g_node_num}")

        self.train_features(v_graph, i)

        return True

    def train_features(self, v_graph, i):
        # Train Node2Vec hidden data
        embed = vite_solver.GraphSolver(dim=self.data_dim)
        embed.build(v_graph)
        embed.train(model='node2vec', num_epoch=2000, resume=False, augmentation_step=1, random_walk_length=40,
                    random_walk_batch_size=100, shuffle_base=1, p=1, q=1, positive_reuse=1,
                    negative_sample_exponent=0.75, negative_weight=5, log_frequency=1000)

        # Extract embedded feature data
        features = np.array(np.copy(embed.vertex_embeddings), dtype=np.float32)

        # Clear memory and data on CPU and GPU
        embed.clear()

        # Pickle hidden feature data
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i, features=True)
        np.save(pickled_filename, features)

    def create_edgelist_from_instance_id(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        # print(f'\tCreating the graph data for instance {instance_id}')
        # Get the cnf file path
        pickled_filename, pickled_folder = self.extract_pickle_filename_and_folder(i)

        if not os.path.exists(pickled_folder):
            os.makedirs(pickled_folder)

        # Load the edgelist data and create sparse matrix
        edgelist_filename = os.path.join(self.root_dir, instance_id + '.edgelist')
        if not os.path.exists(edgelist_filename):
            raise FileNotFoundError(f"Could not find required edgelist file: {edgelist_filename}")

        # Prepare graph for DGL
        # print(f"\tPreparing graph for DGL...")
        edgelist = Edgelist(i)
        edgelist.load_from_file(edgelist_filename)
        graph_adj = SparseMatrix()
        graph_adj.from_edgelist(edgelist)
        g = DGLGraph(graph_adj.data)

        # Pickle loaded data for the next load
        save_graphs(pickled_filename, [g])

    def check_if_pickled(self, i):
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i)
        return os.path.exists(pickled_filename)

    def check_if_pickled_features(self, i):
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i, features=True)
        return os.path.exists(pickled_filename + '.npy')

    def extract_pickle_filename_and_folder(self, i, features=False):
        instance_id: str = self.csv_data_x['instance_id'][i]
        instance_loc = instance_id.split("/" if instance_id.find("/") != -1 else "\\")
        instance_name = instance_loc[-1]
        instance_loc = instance_loc[:-1]
        ext = f".{self.dataset_type}" if features else ".graph"
        pickled_filename = os.path.join(self.csv_x_folder, *instance_loc, instance_name + ext)
        pickled_folder = os.path.dirname(pickled_filename)

        return pickled_filename, pickled_folder

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]

        # Unpickle graph and feature data
        graph = self.load_pickled_graph(i)
        features = self.load_pickled_features(i)

        # Check if we need to re-pickle feature data (if nan had occurred during previous pickling)
        while np.any(np.isnan(features)):
            self.create_node2vec_features(i)
            features = self.load_pickled_features(i)

        graph.ndata['features'] = features

        ys = self.ys[item]

        return graph, ys
