import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from dgl import DGLGraph
from dgl.data import save_graphs, load_graphs
import numpy as np

from src.formats.Edgelist import Edgelist
# from src.formats.Node2Vec import Node2Vec
from src.formats.SparseMatrix import SparseMatrix


class CNFDataset(Dataset):
    def __init__(self, csv_file_x: str, csv_file_y: str, root_dir: str, percent: float = 1.0, return_from_end: bool = False):
        super(CNFDataset).__init__()
        # Checks
        if percent <= 0.0 or percent > 1.0:
            raise ValueError(f"Argument percent must be in range (0.0, 1.0]. You passed: {percent}")

        # Init
        self.graphs = []
        self.ys = []
        self.csv_data_x = pd.read_csv(csv_file_x)
        self.csv_data_y = pd.read_csv(csv_file_y)

        # Create the folder for pickling data
        csv_x = csv_file_x[csv_file_x.rfind(os.sep, 0, -1)+1:-4]
        assert(csv_x != "" and csv_x.find(os.sep) == -1)
        csv_x_folder = os.path.join(os.path.dirname(__file__), "..", "..", "data", csv_x)
        if not os.path.exists(csv_x_folder):
            os.makedirs(csv_x_folder)

        # Load the data
        n = len(self.csv_data_x)
        return_max_num = int(percent*n)
        indices = list(range(return_max_num) if not return_from_end else range(n-1, return_max_num, -1))
        indices.sort()

        # Create and pickle graphs if they aren't pickled before and load ys
        for i in indices:
            # Get the cnf file path
            instance_id: str = self.csv_data_x['instance_id'][i]
            print(f"Checking the data #{i+1}/{len(indices)} for instance id: {instance_id}...")

            # Prepare the folder for pickling
            instance_loc = instance_id.split("/" if instance_id.find("/") != -1 else "\\")
            instance_name = instance_loc[-1]
            instance_loc = instance_loc[:-1]
            pickled_filename = os.path.join(csv_x_folder, *instance_loc, instance_name + '.pickled')
            pickled_folder = os.path.dirname(pickled_filename)

            if not os.path.exists(pickled_folder):
                os.makedirs(pickled_folder)

            pickled = False
            if os.path.exists(pickled_filename):
                pickled = True

            # We first get the true solver runtime data,
            # because we don't pickle that
            ys = self.csv_data_y[self.csv_data_y['instance_id'] == instance_id]
            ys = ys.drop(columns=['instance_id'])
            ys = list(ys.iloc[0])
            self.ys.append(ys)

            if pickled:
                continue

            # Load the edgelist data and create sparse matrix
            edgelist_filename = os.path.join(root_dir, instance_id + '.edgelist')
            if not os.path.exists(edgelist_filename):
                raise FileNotFoundError(f"Could not find required edgelist file: {edgelist_filename}")
            edgelist = Edgelist(i)
            edgelist.load_from_file(edgelist_filename)

            graph_adj = SparseMatrix()
            graph_adj.from_edgelist(edgelist)
            memory_saved = 100 - (graph_adj.data.nnz / graph_adj.data.shape[0]**2 * 100)
            print(f"\tSparseMatrix data shape: {graph_adj.data.shape}. Saved {memory_saved:.2f}% of memory")

            # Create a graph from sparse matrix
            g = DGLGraph(graph_adj.data)
            print(f"\tNumber of nodes: {g.number_of_nodes()}")

            # Populate initial hidden data
            # TODO: Node2Vec is too expensive to calculate, find an alternative
            # node2vec_filename = os.path.join(root_dir, instance_id + '.emb')
            # if not os.path.exists(node2vec_filename):
            #     raise FileNotFoundError(f"Could not find required emb file: {node2vec_filename}")
            # Create a graph
            # node2vec = Node2Vec()
            # node2vec.load_from_file(node2vec_filename)
            # g.ndata['features'] = node2vec.data
            g.ndata['features'] = np.array(np.random.random((g.number_of_nodes(), 2)), dtype=np.float32)
            print(f"\tNode 'features' data shape: {g.ndata['features'].shape}")

            # Pickle loaded data for the next load
            save_graphs(pickled_filename, g)
            print("\tThe graph is pickled for the next load!")

        # Load the pickled graphs
        for i in indices:
            # Get the cnf file path
            instance_id: str = self.csv_data_x['instance_id'][i]
            print(f"Loading the pickled data #{i+1}/{len(indices)} for instance id: {instance_id}...")

            # Prepare the folder for pickling
            instance_loc = instance_id.split("/" if instance_id.find("/") != -1 else "\\")
            instance_name = instance_loc[-1]
            instance_loc = instance_loc[:-1]
            pickled_filename = os.path.join(csv_x_folder, *instance_loc, instance_name + '.pickled')

            g, _ = load_graphs(pickled_filename)
            g = g[0]

            # Add a graph to the list
            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item], self.ys[item]