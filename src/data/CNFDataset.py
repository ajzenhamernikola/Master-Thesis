import os
from torch.utils.data import Dataset
import pandas as pd
from dgl import DGLGraph
import numpy as np

from src.formats.Edgelist import Edgelist
# from src.formats.Node2Vec import Node2Vec
from src.formats.SparseMatrix import SparseMatrix


class CNFDataset(Dataset):
    def __init__(self, csv_file_x, csv_file_y, root_dir):
        super(CNFDataset).__init__()
        self.graphs = []
        self.ys = []
        self.csv_data_x = pd.read_csv(csv_file_x)
        self.csv_data_y = pd.read_csv(csv_file_y)

        # Loading the graphs and ys
        n = len(self.csv_data_x)
        for i in range(n):
            # Get the cnf file path
            instance_id = self.csv_data_x['instance_id'][i]
            print(f"Loading the data for instance id: {instance_id}...")
            # Check if there exist edgelist and node2vec data
            edgelist_filename = os.path.join(root_dir, instance_id + '.edgelist')
            if not os.path.exists(edgelist_filename):
                raise FileNotFoundError(f"Could not find required edgelist file: {edgelist_filename}")
            # node2vec_filename = os.path.join(root_dir, instance_id + '.emb')
            # if not os.path.exists(node2vec_filename):
            #     raise FileNotFoundError(f"Could not find required emb file: {node2vec_filename}")
            # Create a graph
            edgelist = Edgelist(i)
            edgelist.load_from_file(edgelist_filename)
            graph_adj = SparseMatrix()
            graph_adj.from_edgelist(edgelist)
            memory_saved = 100 - (graph_adj.data.nnz / graph_adj.data.shape[0]**2 * 100)
            print(f"\tSparseMatrix data shape: {graph_adj.data.shape}. Saved {memory_saved:.2f}% of memory")
            g = DGLGraph(graph_adj.data)
            print(f"\tNumber of nodes: {g.number_of_nodes()}")
            # Populate initial hidden data
            # TODO: Node2Vec is too expensive to calculate
            # node2vec = Node2Vec()
            # node2vec.load_from_file(node2vec_filename)
            # g.ndata['features'] = node2vec.data
            g.ndata['features'] = np.random.random((g.number_of_nodes(), 2))
            print(f"\tNode 'features' data shape: {g.ndata['features'].shape}")
            # Add a graph to the list
            self.graphs.append(g)

            # Get the true solver runtime data
            ys = self.csv_data_y[self.csv_data_y['instance_id'] == instance_id]
            ys = ys.drop(columns=['instance_id'])
            self.ys.append(ys)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item], self.ys[item]