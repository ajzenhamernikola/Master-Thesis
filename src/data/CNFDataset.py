import os

import numpy as np
import pandas as pd
import dgl
from dgl import DGLGraph
from dgl.data import save_graphs, load_graphs
from torch.utils.data import Dataset
from graphvite import graph as vite_graph
from graphvite import solver as vite_solver

from src.formats.Edgelist import Edgelist
from src.formats.SparseMatrix import SparseMatrix


class CNFDataset(Dataset):
    def __init__(self, csv_file_x: str, csv_file_y: str, root_dir: str, percent: float = 1.0, return_from_end: bool = False):
        super(CNFDataset).__init__()
        # Checks
        if percent <= 0.0 or percent > 1.0:
            raise ValueError(f"Argument percent must be in range (0.0, 1.0]. You passed: {percent}")

        # Init
        self.root_dir = root_dir
        self.ys = []
        self.csv_data_x = pd.read_csv(csv_file_x)
        self.csv_data_y = pd.read_csv(csv_file_y)
        self.embed = vite_solver.GraphSolver(dim=64)
        self.indices = []

        # Create the folder for pickling data
        csv_x = csv_file_x[csv_file_x.rfind(os.sep, 0, -1)+1:-4]
        assert(csv_x != "" and csv_x.find(os.sep) == -1)
        csv_x_folder = os.path.join(os.path.dirname(__file__), "..", "..", "data", csv_x)
        if not os.path.exists(csv_x_folder):
            os.makedirs(csv_x_folder)
        self.csv_x_folder = csv_x_folder

        # Load the data
        n = len(self.csv_data_x)
        return_max_num = int(percent*n)
        indices = list(range(return_max_num) if not return_from_end else range(n-1, return_max_num, -1))
        indices.sort()

        # Keep track of unsuccessfully loaded data, so we can skip them faster
        self.__unsuccessful_txt = os.path.join(os.path.dirname(__file__), "..", "..", "data", "unsuccessful.txt")
        self.__load_already_known_unsuccessful_graphs()

        # Pickle the graphs if they don't exist
        print('\nPickling the data that doesn\'t exist...')
        for i in indices:
            print(f"Checking the pickled state of instance num {i}...")
            # If the data is pickled, then we have everything we need, so save the index
            if self.check_if_pickled(i):
                print(f"\tAlready pickled!")
                self.indices.append(i)
                continue

            # If the graph is known to be unsuccessful, skip it
            if self.__is_unsuccessful_graph(i):
                print(f"\tIs known to be unsuccessful... Skipping this instance!")
                continue

            # Finally, try to pickle data and save the index only if we succeed
            save_indices = self.create_edgelist_from_instance_id(i)
            if save_indices:
                self.indices.append(i)
            else:
                self.__commit_new_unsuccessful_graph(i)

        # Load ys
        for i in self.indices:
            # Get the cnf file path
            instance_id: str = self.csv_data_x['instance_id'][i]
            ys = self.csv_data_y[self.csv_data_y['instance_id'] == instance_id]
            ys = ys.drop(columns=['instance_id'])
            ys = list(ys.iloc[0])
            self.ys.append(ys)

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

    def create_edgelist_from_instance_id(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        print(f'\tCreating the graph data for instance {instance_id}')
        # Get the cnf file path
        pickled_filename, pickled_folder = self.extract_pickle_filename_and_folder(i)

        if not os.path.exists(pickled_folder):
            os.makedirs(pickled_folder)

        # Load the edgelist data and create sparse matrix
        edgelist_filename = os.path.join(self.root_dir, instance_id + '.edgelist')
        if not os.path.exists(edgelist_filename):
            raise FileNotFoundError(f"Could not find required edgelist file: {edgelist_filename}")

        # Prepare graph for DeepWalk
        print(f"\tPreparing graph for DeepWalk...")
        v_graph = vite_graph.Graph()
        v_graph.load(edgelist_filename, as_undirected=False)
        v_graph_node_num = len(v_graph.id2name)

        # Prepare graph for DGL
        print(f"\tPreparing graph for DGL...")
        edgelist = Edgelist(i)
        edgelist.load_from_file(edgelist_filename)
        graph_adj = SparseMatrix()
        graph_adj.from_edgelist(edgelist)
        g = DGLGraph(graph_adj.data)
        g_node_num = g.number_of_nodes()

        # Check if graphs have the same number of nodes
        if v_graph_node_num == g_node_num:
            print(f"\tNumber of nodes: {g_node_num}")
        else:
            print(f"\tMismatching number of nodes: {v_graph_node_num} in graphvite != {g_node_num} in dgl")
            return False

        # Train DeepWalk hidden data
        self.embed.build(v_graph)
        self.embed.train(model='DeepWalk', num_epoch=2000, resume=False, augmentation_step=1, random_walk_length=40,
                         random_walk_batch_size=100, shuffle_base=1, p=1, q=1, positive_reuse=1,
                         negative_sample_exponent=0.75, negative_weight=5, log_frequency=1000)

        # Add hidden feature data
        g.ndata['features'] = np.array(self.embed.vertex_embeddings, dtype=np.float32)
        print(f"\tNode 'features' data shape: {g.ndata['features'].shape}")

        # Clear memory and data on CPU and GPU
        self.embed.clear()

        # Pickle loaded data for the next load
        save_graphs(pickled_filename, g)
        print("\tThe graph is pickled for the next load!")

        return True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        graph = self.load_pickled_graph(i)
        return graph, self.ys[item]

    def check_if_pickled(self, i):
        pickled_filename, _ = self.extract_pickle_filename_and_folder(i)
        return os.path.exists(pickled_filename)

    def extract_pickle_filename_and_folder(self, i):
        instance_id: str = self.csv_data_x['instance_id'][i]
        instance_loc = instance_id.split("/" if instance_id.find("/") != -1 else "\\")
        instance_name = instance_loc[-1]
        instance_loc = instance_loc[:-1]
        pickled_filename = os.path.join(self.csv_x_folder, *instance_loc, instance_name + '.pickled')
        pickled_folder = os.path.dirname(pickled_filename)

        return pickled_filename, pickled_folder
