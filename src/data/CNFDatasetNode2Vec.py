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


def filter_dataset_by_splits(df: pd.DataFrame, splits: list):
    filtered = []

    for i in range(len(df)):
        instance = df.iloc[i]
        filtered.append(instance['split'] in splits)

    return filtered


class CNFDatasetNode2Vec(Dataset):
    def __init__(self, csv_file_x: str, csv_file_y: str, root_dir: str, splits: str):
        super(CNFDatasetNode2Vec).__init__()
        # Checks
        available_splits = ['Train', 'Validation', 'Test']
        splits = splits.split('+')
        for split in splits:
            if split not in available_splits:
                raise ValueError(f'You have passed an unknown split: {splits}. Available values are: {available_splits}')

        # Init
        self.root_dir = root_dir
        self.ys = []
        self.csv_data_x = pd.read_csv(csv_file_x)
        self.csv_data_y = pd.read_csv(csv_file_y)
        self.hidden_features_dim = 64
        self.indices = []
        self.data_dir = "data"
        self.hidden_features_type = "node2vec"
        # Keep track of unsuccessfully loaded data, so we can skip them faster
        self.__unsuccessful_txt = os.path.join(os.path.dirname(__file__), "..", "..", self.data_dir,
                                               f"{self.hidden_features_type}{self.hidden_features_dim}_unsuccessful.txt")
        self.__load_already_known_unsuccessful_graphs()
        print(f"\nPreparing the dataset for phase: {splits}")

        # Keeping only the data required for the current split
        self.csv_data_x = self.csv_data_x[filter_dataset_by_splits(self.csv_data_x, splits)]
        self.csv_data_x.index = pd.Series(range(len(self.csv_data_x)))

        # Create the folder for pickling data
        csv_x_folder = os.path.join(os.path.dirname(__file__), "..", "..", self.data_dir)
        if not os.path.exists(csv_x_folder):
            os.makedirs(csv_x_folder)
        self.csv_x_folder = csv_x_folder

        # Load the data
        indices = list(self.csv_data_x.index)

        # percent = 0.1
        # indices = indices[:int(percent*len(indices))]

        # Pickle the graphs if they don't exist
        print('\nPickling the graph data that doesn\'t exist...')
        for i in indices:
            print(f"(Graph) Checking the pickled state of instance num {i+1}/{len(indices)}...")

            # Skip unsolvable indices
            if not self.is_solvable(i):
                # print("\tNonsolvable - skipping")
                self.__commit_new_unsuccessful_graph(i)
                continue

            if self.check_if_pickled(i):
                # print(f"\tAlready pickled!")
                continue

            # Pickle graph data
            self.create_edgelist_from_instance_id(i)

        wrong_instances = []

        # Pickle the features if they don't exist
        print('\nPickling the feature data that doesn\'t exist...')
        for i in indices:
            # print(f"(Features) Checking the pickled state of instance num {i}...")

            # Skip unsolvable indices
            if self.__is_unsuccessful_graph(i):
                # print("\tNonsolvable: skipping...")
                continue

            # If the data is pickled, then we have everything we need, so save the index
            if self.check_if_pickled_features(i):
                # print(f"\tAlready pickled!")
                self.indices.append(i)
                continue

            # Finally, try to pickle feature data and save the index
            try:
                self.create_node2vec_features(i)
                self.indices.append(i)
            except ValueError as e:
                instance_id: str = self.csv_data_x['instance_id'][i]
                edgelist_filename = os.path.join(self.root_dir, instance_id + '.edgelist')
                wrong_instances.append(edgelist_filename)
                print(e)

        wrong_instances_filename = f"edgelist_wrong_{'+'.join(splits)}.txt"
        if len(wrong_instances) > 0:
            with open(wrong_instances_filename, "w") as file:
                file.write("\n".join(wrong_instances))
        else:
            if os.path.exists(wrong_instances_filename):
                os.remove(wrong_instances_filename)

        # Load ys
        for i in self.indices:
            # Get the cnf file path
            ys = self.get_ys(i)
            ys = log10_transform_data(ys)
            self.ys.append(ys)

        print(f"\nNumber of instances in this dataset is: {len(self.indices)}")

    def get_ys(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        ys = self.csv_data_y[self.csv_data_y['instance_id'] == instance_id]
        return np.array(ys.drop(columns=['instance_id']).iloc[0], dtype=np.float32)

    def is_solvable(self, i):
        ys = self.get_ys(i)
        return not np.all(ys == 1200.0)

    def __commit_new_unsuccessful_graph(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        self.__unsuccessful_indices.append(instance_id)
        with open(self.__unsuccessful_txt, 'w') as f:
            sorted_instances = list(set(map(lambda x: x.strip() + '\n', self.__unsuccessful_indices)))
            sorted_instances.sort()
            for inst in sorted_instances:
                f.write(inst)

    def __load_already_known_unsuccessful_graphs(self):
        if not os.path.exists(self.__unsuccessful_txt):
            self.__unsuccessful_indices = []
            return

        with open(self.__unsuccessful_txt, 'r') as f:
            self.__unsuccessful_indices = list(set(map(lambda x: x.strip(), f.readlines())))

    def __is_unsuccessful_graph(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        return self.__unsuccessful_indices.count(instance_id) > 0

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
            raise ValueError(f"\tMismatching number of nodes: {v_graph_node_num} in graphvite != {g_node_num} in dgl.")

        # print(f"\tNumber of nodes: {g_node_num}")

        self.train_features(v_graph, i)

    def train_features(self, v_graph, i):
        # Train Node2Vec hidden data
        embed = vite_solver.GraphSolver(dim=self.hidden_features_dim)
        embed.build(v_graph)
        embed.train(model=self.hidden_features_type, num_epoch=2000, resume=False, augmentation_step=1,
                    random_walk_length=40, random_walk_batch_size=100, shuffle_base=1, p=1, q=1, positive_reuse=1,
                    negative_sample_exponent=0.75, negative_weight=5, log_frequency=1000)

        # Extract embedded feature data
        sorted_features = np.empty(embed.vertex_embeddings.shape, dtype=np.float32)
        id2name = list(map(lambda x: int(x), v_graph.id2name))
        for j in range(embed.vertex_embeddings.shape[0]):
            sorted_features[id2name[j], :] = embed.vertex_embeddings[j]

        # Clear memory and data on CPU and GPU
        embed.clear()

        # Pickle hidden feature data
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i, features=True)
        np.save(pickled_filename, sorted_features)

    def create_edgelist_from_instance_id(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        # print(f'\tCreating the graph data for instance {instance_id}')
        # Get the cnf file path
        pickled_filename, pickled_folder = self.extract_pickle_filename_and_folder(i)

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
        ext = f".{self.hidden_features_type}{self.hidden_features_dim}" if features else ".graph"
        pickled_filename = os.path.join(self.csv_x_folder, *instance_loc, instance_name + ext)
        pickled_folder = os.path.dirname(pickled_filename)
        if os.path.exists(pickled_folder) and not os.path.isdir(pickled_folder):
            os.remove(pickled_folder)
        if not os.path.exists(pickled_folder):
            os.makedirs(pickled_folder)

        return os.path.abspath(pickled_filename), os.path.abspath(pickled_folder)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        
        # Unpickle graph and feature data
        graph = self.load_pickled_graph(i)
        
        degrees = False
        if degrees:
            features = graph.out_degrees().reshape((-1, 1))
        else:
            features = self.load_pickled_features(i)

            # Check if we need to re-pickle feature data (if nan had occurred during previous pickling)
            while np.any(np.isnan(features)):
                self.create_node2vec_features(i)
                features = self.load_pickled_features(i)

        graph.ndata['features'] = features

        ys = self.ys[item]

        return graph, ys
