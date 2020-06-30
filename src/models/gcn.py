import os

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from dgl.nn.pytorch import GraphConv, SumPooling, MaxPooling, AvgPooling
from sklearn import metrics
from torch.utils.data import DataLoader

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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation, dropout_p, pooling="avg"):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))

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
        # Remember the last conv. representations, in case we want them
        g.ndata['h'] = h

        # Perform pooling over all nodes in each graph in every layer
        pooled_h = self.pool(g, h)
        # Return the predicted data in linear layer over pooled data
        linear = self.linear(pooled_h)

        return linear


# Train the model
def train(device):
    # Load train data
    csv_file_x = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'max_vars_5000_max_clauses_200000_top_500.csv')
    csv_file_y = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'all_data_y.csv')
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'INSTANCES')
    trainset = CNFDataset(csv_file_x, csv_file_y, root_dir, 0.7)
    print("\nLoaded the CNFDataset!\n")

    data_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate(device))
    print("\nCreated the data loader!\n")

    # Create model
    input_dim = 2
    hidden_dim = 50
    output_dim = 31
    num_layers = 5
    activation = f.relu
    dropout_p = 0.3
    pooling = "avg"
    model = Regressor(input_dim,
                      hidden_dim,
                      output_dim,
                      num_layers,
                      activation,
                      dropout_p,
                      pooling)
    model.to(device)
    loss_func = nn.MSELoss()
    learning_rate = 1e-4
    weight_decay = 0.3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    print("\nModel created! Training...\n")

    # Start training
    epochs = 200
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        iter_idx = -1
        for iter_idx, (bg, label) in enumerate(data_loader):
            # Predict
            prediction = model(bg.to(device))
            loss = loss_func(prediction, label.to(device))
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Report loss
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter_idx + 1)
        print(f'Epoch {epoch}, loss {epoch_loss}')
        epoch_losses.append(epoch_loss)
        # Stop if there isn't substantial progress in loss over the last 10 epochs
        if epoch >= 10:
            avg_loss = np.average(epoch_losses[epoch-10:epoch])
            last_loss = epoch_loss
            one_percent_progress = 0.01 * avg_loss
            if np.fabs(last_loss - avg_loss) < one_percent_progress:
                print(f"\nStopping the training due to low progress over the last 10 epochs:")
                print(f"\tLast loss: {last_loss}, Average loss: {avg_loss}")
                break

    # Serialize model for later usage
    torch.save([model, epochs], os.path.join(os.path.dirname(__file__), '..', '..', 'models',
               f'gcn_model_{hidden_dim}_{num_layers}_{dropout_p}_{pooling}_{learning_rate}_{weight_decay}_{epochs}'))


# Test the model
def test():
    # TODO: Load test data
    csv_file_x = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'max_vars_5000_max_clauses_200000_top_500.csv')
    csv_file_y = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'INSTANCES', 'chosen_data', 'all_data_y.csv')
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'INSTANCES')
    testset = CNFDataset(csv_file_x, csv_file_y, root_dir, 0.3, True)

    test_graph_list, y_true = testset.graphs, np.array(testset.ys)

    # Load the model
    model_name = 'gcn_model_50_5_0.3_avg_0.0001_0.3_200'
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', model_name)
    data = torch.load(model_path)
    model = data[0]
    epochs = data[1]
    model.eval()
    y_pred = np.empty((0, 31))

    # Predict
    print("\nPredicting...\n")
    pred_data_name = model_path + '.predicted'
    # Check if there are predicted data for this model, and load it
    if os.path.exists(pred_data_name):
        y_pred = np.load(pred_data_name)
    # Otherwise, predict the values and save to file
    else:
        with torch.no_grad():
            for i, (graph, true_y) in enumerate(zip(test_graph_list, y_true)):
                pred_y = model(graph)
                pred_y = np.array(pred_y)
                y_pred = np.vstack((y_pred, pred_y))
            # Save the predicted data
            np.save(pred_data_name, y_pred)

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

    train_not_test = False  # To save memory
    if train_not_test:
        train(device)
    else:
        test()
