import os
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import GraphConv, SumPooling, MaxPooling, AvgPooling
from sklearn import metrics
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from src.data.CNFDatasetNode2Vec import CNFDatasetNode2Vec

DatasetClass = CNFDatasetNode2Vec
csv_file_x = os.path.join(os.path.dirname(__file__),
                          '..', '..', 'INSTANCES', 'chosen_data', 'max_vars_5000_max_clauses_200000.csv')
csv_file_y = os.path.join(os.path.dirname(__file__),
                          '..', '..', 'INSTANCES', 'chosen_data', 'all_data_y.csv')
root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'INSTANCES')


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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation, activation_params, dropout_p,
                 pooling="avg"):
        super(Regressor, self).__init__()

        # Checks
        if num_layers < 2:
            raise ValueError(f"Argument num_layers must be >= 2. You passed {num_layers}")

        if activation == "relu":
            self.activation = nn.ReLU(**activation_params)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(**activation_params)
        elif activation == "elu":
            self.activation = nn.ELU(**activation_params)
        else:
            raise NotImplementedError(f"Unknown activation method: {pooling}")

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphConv(input_dim, hidden_dim))
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))

        # Additional layers
        if pooling == "avg":
            self.pool = AvgPooling()
        elif pooling == "sum":
            self.pool = SumPooling()
        elif pooling == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError(f"Unknown pooling method: {pooling}")

        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, g: dgl.DGLGraph):
        # Use the already set features data as initial node features
        h = g.ndata['features']
        # Apply layers
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            h = self.activation(h)

        # Perform pooling over all nodes in each graph in every layer
        pooled_h = self.pool(g, h)

        # Return the predicted data in linear layer over pooled data
        linear = self.linear(pooled_h)

        return linear


# Train the model
def train(train_device):
    # Load train data
    trainset = DatasetClass(csv_file_x, csv_file_y, root_dir, "train")
    data_loader_train = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate(train_device))

    # Load val data
    valset = DatasetClass(csv_file_x, csv_file_y, root_dir, "val")
    data_loader_val = DataLoader(valset, batch_size=1, shuffle=True, collate_fn=collate(train_device))

    # Model params
    dataset = trainset.dataset_type
    input_dim = trainset.data_dim
    hidden_dim = 64
    output_dim = 31
    num_layers = 2
    activation = "elu"
    activation_params = {"alpha": 0.3}
    dropout_p = 0.2
    pooling = "avg"
    # Optimizer params
    learning_rate = 1e-4
    weight_decay = 0.0
    # Num of epochs
    epochs = 200

    # Model name
    mfn = f'gcn_model_{hidden_dim}_{num_layers}_{dropout_p}_{pooling}_{learning_rate}_{weight_decay}_{epochs}_{dataset}'
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', mfn)
    if os.path.exists(model_path):
        print("\nModel had already been trained!")
        return model_path

    # Create model
    model = Regressor(input_dim,
                      hidden_dim,
                      output_dim,
                      num_layers,
                      activation,
                      activation_params,
                      dropout_p,
                      pooling)
    model.to(train_device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("\nModel created! Training...\n")

    # Start training
    train_losses = []
    val_losses = []
    train_times = []
    val_times = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        iter_idx = -1

        time_start = time.process_time_ns()
        for iter_idx, (bg, label) in enumerate(data_loader_train):
            # Predict
            prediction = model(bg.to(train_device))
            loss = loss_func(prediction, label.to(train_device))
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Report loss
            train_loss += loss.detach().item()
        time_elapsed = time.process_time_ns() - time_start
        train_times.append(time_elapsed)

        train_loss /= (iter_idx + 1)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0

        time_start = time.process_time_ns()
        for iter_idx, (bg, label) in enumerate(data_loader_val):
            # Predict
            prediction = model(bg.to(train_device))
            loss = loss_func(prediction, label.to(train_device))
            val_loss += loss.detach().item()
        time_elapsed = time.process_time_ns() - time_start
        val_times.append(time_elapsed)

        val_loss /= (iter_idx + 1)
        val_losses.append(val_loss)

        print(f'\nEpoch {epoch}, train loss {train_loss}, val loss {val_loss}')

    # Serialize model for later usage
    torch.save([model, train_losses, val_losses, train_times, val_times], model_path)
    return model_path


# Test the model
def test(predict_device, test_device, model_path):
    testset = DatasetClass(csv_file_x, csv_file_y, root_dir, "test")
    data_loader_test = DataLoader(testset, batch_size=1, shuffle=True, collate_fn=collate(predict_device))

    # Load the model
    data = torch.load(model_path)
    model = data[0]
    model.to(predict_device)
    train_losses = data[1]
    val_losses = data[2]
    plt.plot(range(len(train_losses)), train_losses, 'b-')
    plt.plot(range(len(val_losses)), val_losses, 'r-')
    plt.show()

    # Prepare for predicting
    model.eval()
    y_pred = np.empty((0, 31))
    y_true = np.empty((0, 31))

    # Predict
    print("\nPredicting...\n")
    pred_data_name = model_path + '.predicted'
    true_data_name = model_path + '.true'
    # Check if there are predicted data for this model, and load it
    pred_exists = os.path.exists(pred_data_name)
    true_exists = os.path.exists(true_data_name)
    if pred_exists and true_exists:
        y_pred = np.load(pred_data_name)
        y_true = np.load(true_data_name)
    # Otherwise, predict the values and save to file
    else:
        with torch.no_grad():
            for i, (graph, true_y) in enumerate(data_loader_test):
                pred_y = model(graph.to(predict_device))
                pred_y = np.array(pred_y.to(test_device))
                y_pred = np.vstack((y_pred, pred_y))
                y_true = np.vstack((y_true, true_y.to(test_device)))
            # Save the predicted data
            np.save(pred_data_name, y_pred)
            np.save(true_data_name, y_true)

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

    # More details
    train_times = data[3]
    val_times = data[4]
    print(f"Average training time: {np.average(train_times) / 1000*1000*1000}s")
    print(f"Average validating time: {np.average(val_times) / 1000*1000*1000}s")


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_device = torch.device("cpu")

    model_path = train(device)
    test(device, test_device, model_path)
