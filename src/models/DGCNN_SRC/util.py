import argparse
import os
import pickle as pkl
from timeit import default_timer as timer
import math

import networkx as nx
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
import graphvite

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0,
                     help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=False,
                     help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]


class GNNGraph(object):
    def __init__(self, g, labels, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
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


def create_node2vec_features(data_dir: str, instance_id: str):
    edgelist_filename = os.path.join(data_dir, instance_id + '.edgelist')

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
    pickled_filename = os.path.join(data_dir, "..", "data", instance_id + '.node2vec64')
    np.save(pickled_filename, sorted_features)


def pickle_data(data_dir: str, instance_ids: list, splits: dict):
    print('Pickling data')

    time_start = timer()

    n_g = len(instance_ids)
    pbar = tqdm(range(n_g), unit='graph')
    metadata_filename = os.path.join(data_dir, "DGCNN", "dgcnn_parsing_metadata.pickled")
    os.makedirs(os.path.dirname(metadata_filename), exist_ok=True)

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
        pickle_file = os.path.join(data_dir, instance_id + ".dgcnn.pickled")
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)

        if os.path.exists(pickle_file):
            continue

        graph_filename = os.path.join(data_dir, instance_id + ".dgcnn.txt")
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
            features_filename = os.path.join(data_dir, "..", "data", instance_id + '.node2vec64.npy')
            if not os.path.exists(features_filename):
                create_node2vec_features(data_dir, instance_id)
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

    time_elapsed = timer() - time_start
    print(f"Data pickled in {time_elapsed:.2f}s\n")

    print("Instances distribution:")
    print(f"\tTrain: {splits['Train']}\n\tValidation: {splits['Validation']}\n\tTest: {splits['Test']}\n")

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    # Set data parameters
    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted(num_nodes_l)
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    cmd_args.num_class = num_class
    cmd_args.feat_dim = len(feat_dict)  # maximum node label (tag)
    cmd_args.edge_feat_dim = 0
    cmd_args.attr_dim = 64
    print(f'K used in SortPooling is: {cmd_args.sortpooling_k}')


def load_next_batch(data_dir, instance_ids, selected_idx, splits, dataset_type):
    train_size = splits["Train"]
    val_size = splits["Validation"]

    if dataset_type == "Train":
        instance_ids = instance_ids[:train_size]
    elif dataset_type == "Validation":
        instance_ids = instance_ids[train_size:train_size + val_size]
    elif dataset_type == "Train+Validation":
        instance_ids = instance_ids[:train_size + val_size]
    elif dataset_type == "Test":
        instance_ids = instance_ids[train_size + val_size:]

    batch_graph = []
    labels = []
    for idx in selected_idx:
        instance_id = instance_ids[idx]
        pickle_file = os.path.join(data_dir, instance_id + ".dgcnn.pickled")
        with open(pickle_file, "rb") as f:
            batch_graph.append(pkl.load(f))
            labels.append(batch_graph[-1].labels)

    return batch_graph, labels


def loop_dataset(data_dir, instance_ids, splits, epoch, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, dataset_type="Train"):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph, targets = load_next_batch(data_dir, instance_ids, selected_idx, splits, dataset_type)
        all_targets += targets

        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            predicted = pred.cpu().detach()
            all_scores.append(predicted)  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    np.savetxt(f'models/DGCNN/{dataset_type}_{epoch}_scores.txt', all_scores)  # output predictions

    if not classifier.regression and cmd_args.printAUC:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0]))

    return avg_loss


def time_for_early_stopping(val_losses: list, look_behind: int):
    if len(val_losses) < look_behind:
        return False

    last_epoch_loss = val_losses[-1]
    avg_epoch_loss = np.average(val_losses[-look_behind:])

    # Stop training if the progress in last epoch is less than 7.5% of average losses
    return avg_epoch_loss - last_epoch_loss < 0.075 * avg_epoch_loss
