import os
import sys
import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np

from src.data.CNFDataset import CNFDataset


def collate(dev):
    def collate_fn(samples):
        """
            Forms a mini-batch from a given list of graphs and label pairs
            :param samples: list of tuple pairs (graph, label)
            :return:
            """
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, device=dev, dtype=torch.float32)
    return collate_fn


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_predicted_vals, num_layers, activation, dropout_p):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        # Output layer
        self.regress = nn.Linear(hidden_dim, n_predicted_vals)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, g: dgl.DGLGraph):
        # Use the already set features data as initial node features
        h = g.ndata['features']
        # Apply layers
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        g.ndata['h'] = h
        # Global average pooling over hidden feature data
        hg = dgl.mean_nodes(g, 'h')
        # Return the predicted data in linear layer over pooled data
        return self.regress(hg)


# Train the model
def train(device):
    # Load train data
    csv_file_x = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'max_vars_5000_max_clauses_200000_top_500.csv')
    csv_file_y = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'all_data_y.csv')
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'INSTANCES')
    trainset = CNFDataset(csv_file_x, csv_file_y, root_dir)
    print("\nLoaded the CNFDataset!\n")

    data_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate(device))
    print("\nCreated the data loader!\n")

    # Create model
    num_predicted_values = 31
    model = Regressor(in_dim=2,
                      hidden_dim=256,
                      n_predicted_vals=num_predicted_values,
                      num_layers=2,
                      activation=f.relu,
                      dropout_p=0.1)
    model.to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-3)
    model.train()
    print("\nModel created! Training...\n")

    # Start training
    epochs = 100
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        iter_idx = -1
        for iter_idx, (bg, label) in enumerate(data_loader):
            prediction = model(bg.to(device))
            loss = loss_func(prediction, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter_idx + 1)
        print(f'Epoch {epoch}, loss {epoch_loss}')
        epoch_losses.append(epoch_loss)
    torch.save([model, epochs], os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gcn_model'))


# Test the model
def test():
    # TODO: Load test data
    csv_file_x = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'max_vars_5000_max_clauses_200000_top_500.csv')
    csv_file_y = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'all_data_y.csv')
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'INSTANCES')
    trainset = CNFDataset(csv_file_x, csv_file_y, root_dir)

    test_graph_list, y_true = trainset.graphs, np.array(trainset.ys)

    # Load the model
    data = torch.load(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'gcn_model'))
    model = data[0]
    epochs = data[1]
    model.eval()
    y_pred = np.empty((0, 31))

    # Predict
    print("\nPredicting...\n")
    with torch.no_grad():
        for i, (graph, true_y) in enumerate(zip(test_graph_list, y_true)):
            pred_y = model(graph)
            pred_y = np.array(pred_y)
            y_pred = np.vstack((y_pred, pred_y))

    # Evaluate
    print("\nEvaluating...\n")
    r2_scores_test = np.empty((31,))
    rmse_scores_test = np.empty((31,))
    for i in range(31):
        r2_scores_test[i] = metrics.r2_score(y_true[:, i:i + 1], y_pred[:, i:i + 1])
        rmse_scores_test[i] = metrics.mean_squared_error(y_true[:, i:i + 1], y_pred[:, i:i + 1], squared=False)

    r2_score_test_avg = np.average(r2_scores_test)
    rmse_score_test_avg = np.average(rmse_scores_test)
    print(f'Average R2 score: {r2_score_test_avg}, Average RMSE score: {rmse_score_test_avg}')


if __name__ == "__main__":
    # Set the device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    # Start training
    # train(device)

    test()
