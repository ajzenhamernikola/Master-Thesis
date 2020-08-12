import os
from timeit import default_timer as timer
import math
from typing import Union

import pandas as pd
import pickle as pkl
import numpy as np
import graphvite
from tqdm import tqdm
import networkx as nx

from ..parsers.libparsers import parserslib
from ..os.process import start_process
from ..cnf.process_cnf_attributes import sort_by_split, is_unsolvable
from ..algorithms.math import log10_transform_data
from ..cnf.CNFDatasetNode2Vec import CNFDatasetNode2Vec


IntOrFloat = Union[int, float]


class GNNGraph(object):
    def __init__(self, g, labels, node_tags=None, node_features=None):
        """
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        """
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.labels = labels
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


def generate_satzilla_features(csv_filename, root_dirname, cnf_dir):
    data = pd.read_csv(csv_filename)
    idx = 0
    for filename in data['instance_id']:
        idx += 1
        features_filename = os.path.join(cnf_dir, filename + '.features')
        if os.path.exists(features_filename):
            continue
        print('Creating SATzilla2012 features for file #{0}: {1}...'.format(idx, filename))
        start_process(f'{root_dirname}/third-party/SATzilla2012_features/features', [filename, features_filename])


def generate_edgelist_formats(csv_filename, directory='.'):
    data = pd.read_csv(csv_filename)
    idx = 0
    for filename in data['instance_id']:
        filename = os.path.join(os.path.abspath(directory), filename)
        idx += 1
        features_filename = filename + '.edgelist'
        if os.path.exists(features_filename):
            continue
        print('\nCreating Edgelist format for file #{0}: {1}...'.format(idx, filename))
        basedir = os.path.dirname(filename)
        filepath = os.path.basename(filename)
        parserslib.parse_dimacs_to_edgelist(basedir, filepath)


def generate_dgcnn_formats(csv_filename, csv_labels, cnf_dir, model_output_dir, model):
    data = pd.read_csv(csv_filename).sort_values(by="split", key=sort_by_split)
    ys = pd.read_csv(csv_labels)
    
    features_filenames = []
    instance_ids = []
    splits = {}
    
    ys_by_splits = {}
    instances_by_splits = {}
    
    for i in range(len(data)):
        instance_id = data.iloc[i]['instance_id']
        if is_unsolvable(ys, instance_id):
            continue

        split = data.iloc[i]['split']
        if split not in splits:
            splits[split] = 0
        splits[split] += 1

        if split == "None":
            continue

        instance_ids.append(instance_id)
        labels = log10_transform_data(ys[ys["instance_id"] == instance_id].drop(columns="instance_id").values[0]).reshape(1, -1)
        
        if split not in ys_by_splits:
            ys_by_splits[split] = []
        if split not in instances_by_splits:
            instances_by_splits[split] = []
        
        ys_by_splits[split].append(labels)
        instances_by_splits[split].append(instance_id)

        filename = os.path.join(cnf_dir, instance_id)
        features_filename = filename + '.dgcnn.txt'
        features_filenames.append(features_filename)
        if os.path.exists(features_filename):
            continue

        print('\nCreating DGCNN format for file #{0}: {1}...'.format(i + 1, filename))
        basedir = os.path.dirname(filename)
        filepath = os.path.basename(filename)
        parserslib.parse_dimacs_to_dgcnn_vcg(basedir, filepath, " ".join([str(l) for l in labels]))

    print(f"Found:\n\t{splits['Train']} in Train\n\t{splits['Validation']} in Validation\n\t{splits['Test']} in Test" +
          f"\n\t{splits['None']} in None")

    # Saving true labels
    for split in splits.keys():
        if split in ys_by_splits:
            labels = ys_by_splits[split]
            np.savetxt(os.path.join(model_output_dir, model, f"{split}_ytrue.txt"), 
                       np.array(labels, dtype=np.float32).reshape((-1, 31)))
        if split in instances_by_splits:
            with open(os.path.join(model_output_dir, model, f"{split}_ytrue_instances.txt"), "w") as f:
                f.writelines(i + '\n' for i in instances_by_splits[split])

    # Saving parsed instance ids
    instance_ids_file = os.path.join(model_output_dir, model, "instance_ids.pickled")
    with open(instance_ids_file, "wb") as instance_ids_file:
        pkl.dump([instance_ids, splits], instance_ids_file)


def create_node2vec_features(cnf_dir: str, instance_id: str):
    edgelist_filename = os.path.join(cnf_dir, instance_id + '.edgelist')

    # Prepare graph for Node2Vec
    v_graph = graphvite.graph.Graph()
    v_graph.load(edgelist_filename, as_undirected=False)

    # Train Node2Vec hidden data
    embed = graphvite.solver.GraphSolver(dim=64)
    embed.build(v_graph)
    embed.train(model="node2vec", num_epoch=2000, resume=False, augmentation_step=1,
                random_walk_length=40, random_walk_batch_size=100, shuffle_base=1, p=1, q=1, positive_reuse=1,
                negative_sample_exponent=0.75, negative_weight=5, log_frequency=1000)

    # Extract embedded feature data
    sorted_features = np.empty(embed.vertex_embeddings.shape, dtype=np.float32)
    id2name = list(map(lambda x: int(x), v_graph.id2name))
    try:
        for j in range(embed.vertex_embeddings.shape[0]):
            sorted_features[id2name[j], :] = embed.vertex_embeddings[j]
    except IndexError as e:
        print(embed.vertex_embeddings.shape)
        print(sorted_features.shape)
        print(e)
        exit(1)

    # Clear memory and data on CPU and GPU
    embed.clear()

    # Pickle hidden feature data
    pickled_filename = os.path.join(cnf_dir, instance_id + '.node2vec64')
    np.save(pickled_filename, sorted_features)


def generate_dgcnn_pickled_data(model_output_dir: str, cnf_dir: str, instance_ids: list, splits: dict, sortpooling_k: IntOrFloat):
    print('Pickling data')

    time_start = timer()

    n_g = len(instance_ids)
    pbar = tqdm(range(n_g), unit='graph')
    metadata_filename = os.path.join(model_output_dir, "DGCNN", "dgcnn_parsing_metadata.pickled")

    if os.path.exists(metadata_filename):
        with open(metadata_filename, "rb") as meta_f:
            metadata = pkl.load(meta_f)
            feat_dict = metadata[0]
            num_nodes_l = metadata[2]
    else:
        feat_dict = {}
        num_nodes_l = []

    for i in pbar:
        pbar.set_description(f"Loading graph instance #{i} of {n_g}")
        instance_id = instance_ids[i]

        # Check if a graph is already pickled
        pickle_file = os.path.join(cnf_dir, instance_id + ".dgcnn.pickled")
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)

        if os.path.exists(pickle_file):
            continue

        graph_filename = os.path.join(cnf_dir, instance_id + ".dgcnn.txt")
        with open(graph_filename, "r") as f:
            # Load graph data
            l = []
            n = 0
            row = f.readline().strip().split()
            for row_i, data in enumerate(row):
                if row_i == 0:
                    n = int(data)
                else:
                    l.append(np.float(data))
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [np.int32(w) for w in row]

                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            assert len(g) == n

            # Load feature data
            features_filename = os.path.join(cnf_dir, instance_id + '.node2vec64.npy')
            if not os.path.exists(features_filename):
                create_node2vec_features(cnf_dir, instance_id)
            node_features = np.load(features_filename)

            # Create the graph
            gnn_graph = GNNGraph(g, l, node_tags, node_features)
            num_nodes_l.append(gnn_graph.num_nodes)

            # Pickle the graph for next loading
            with open(pickle_file, "wb") as pickle_f:
                pkl.dump(gnn_graph, pickle_f)

            # Pickle the current version of metadata
            with open(metadata_filename, "wb") as meta_f:
                metadata = [feat_dict, len(l), num_nodes_l]
                pkl.dump(metadata, meta_f)

    # Load the metadata
    with open(metadata_filename, "rb") as meta_f:
        metadata = pkl.load(meta_f)
        feat_dict = metadata[0]
        num_class = metadata[1]
        num_nodes_l = metadata[2]
        feat_dim = len(feat_dict)

    time_elapsed = timer() - time_start
    print(f"Data pickled in {time_elapsed:.2f}s\n")

    print("Instances distribution:")
    print(f"\tTrain: {splits['Train']}\n\tValidation: {splits['Validation']}\n\tTest: {splits['Test']}\n")

    print('# classes: %d' % num_class)
    print('# maximum node tag: %d' % feat_dim)

    # Set data parameters
    if sortpooling_k <= 1:
        num_nodes_list = sorted(num_nodes_l)
        sortpooling_k = num_nodes_list[int(math.ceil(sortpooling_k * len(num_nodes_list))) - 1]
        sortpooling_k = max(10, sortpooling_k)

    edge_feat_dim = 0
    attr_dim = 64
    print(f'K used in SortPooling is: {sortpooling_k}')

    return num_class, feat_dim, edge_feat_dim, attr_dim, sortpooling_k


def generate_cnf_datasets_for_training(root_dir, csv_file_x, csv_file_y):
    trainset = CNFDatasetNode2Vec(csv_file_x, csv_file_y, root_dir, "Train")
    valset = CNFDatasetNode2Vec(csv_file_x, csv_file_y, root_dir, "Validation")
    trainvalset = CNFDatasetNode2Vec(csv_file_x, csv_file_y, root_dir, "Train+Validation")
    return trainset, valset, trainvalset


def generate_cnf_datasets_for_testing(root_dir, csv_file_x, csv_file_y):
    return CNFDatasetNode2Vec(csv_file_x, csv_file_y, root_dir, "Test")


def generate_cnf_datasets(root_dir, csv_file_x, csv_file_y):
    return generate_cnf_datasets_for_training(root_dir, csv_file_x, csv_file_y), \
           generate_cnf_datasets_for_testing(root_dir, csv_file_x, csv_file_y)
