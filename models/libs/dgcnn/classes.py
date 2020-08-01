import os
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim

from .dgcnn_embedding import DGCNN
from .mlp_dropout import MLPClassifier, MLPRegression
from ..features import prepare_feature_labels


IntOfFloat = Union[int, float]


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden: int, num_class: int, dropout: bool,
                 feat_dim: int, attr_dim: int, edge_feat_dim: int,
                 sortpooling_k: IntOfFloat, conv1d_activation: str, mode: str, regression=True):
        super(Predictor, self).__init__()

        self.feat_dim = feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.mode = mode
        self.regression = regression
        self.gnn = DGCNN(latent_dim=latent_dim,
                         output_dim=out_dim,
                         num_node_feats=feat_dim + attr_dim,
                         num_edge_feats=edge_feat_dim,
                         k=sortpooling_k,
                         conv1d_activation=conv1d_activation)

        out_dim = out_dim
        if out_dim == 0:
            out_dim = self.gnn.dense_dim

        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=hidden, output_size=num_class,
                                     with_dropout=dropout)
        else:
            self.mlp = MLPClassifier(input_size=out_dim, hidden_size=hidden, num_class=num_class,
                                     with_dropout=dropout)

    def forward(self, batch_graph):
        embed, labels = self.output_features(batch_graph)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        node_feat = None
        edge_feat = None
        labels = None
        feature_label = prepare_feature_labels(batch_graph, self.feat_dim, self.edge_feat_dim, self.mode,
                                               self.regression)

        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label

        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels


class DGCNNPredictor(object):
    def __init__(self, cnf_dir: str, model_output_dir: str, model: str,
                 instance_ids: list, splits: dict,
                 latent_dim: int, out_dim: int, hidden: int, num_class: int, dropout: bool,
                 feat_dim: int, attr_dim: int, edge_feat_dim: int,
                 sortpooling_k: IntOfFloat, conv1d_activation: str, learning_rate: float,
                 mode: str, regression=True):
        # Inits
        self.predictor = None
        self.cnf_dir = cnf_dir
        self.model_output_dir = model_output_dir
        self.model = model
        self.model_filename = os.path.join(model_output_dir, model, "best_DGCNN_model")
        self.instance_ids = instance_ids
        self.splits = splits
        self.feat_dim = feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.mode = mode
        self.regression = regression

        # Create a predictor
        overwrite = True
        if os.path.exists(self.model_filename):
            print("\nModel already exist. Would you like to overwrite it [o], continue training [t], " +
                  "or use existing [e]?")
            response = input().lower()
            if response == "e":
                return
            if response == "t":
                overwrite = False
            elif response != "o":
                raise ValueError(f"Unknown response: '{response}'. You must enter: 'o', 't' or 'e'")

        if overwrite:
            self.predictor = Predictor(latent_dim, out_dim, hidden, num_class, dropout,
                                       feat_dim, attr_dim, edge_feat_dim,
                                       sortpooling_k, conv1d_activation, mode, regression)
        else:
            self.load()

        if mode == 'gpu':
            self.predictor = self.predictor.cuda()
            print("Optimizing on a GPU\n")
        else:
            print("Optimizing on a CPU\n")

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def persist(self):
        torch.save(self.predictor, self.model_filename)

    def load(self):
        self.predictor = torch.load(self.model_filename)
