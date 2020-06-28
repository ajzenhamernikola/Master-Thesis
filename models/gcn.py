import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np


def collate(samples):
    """
    Forms a mini-batch from a given list of graphs and label pairs
    :param samples: list of tuple pairs (graph, label)
    :return:
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_predicted_vals):
        super(Regressor, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.regress = nn.Linear(hidden_dim, n_predicted_vals)

    def forward(self, g: dgl.DGLGraph):
        # Use the node2vec data as initial node features
        h = g.ndata['node2vec']
        h = f.relu(self.conv1(g, h))
        h = f.relu(self.conv2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        return self.regress(hg)


# Load the data
train_graph_list, train_label_dict = load_graphs("./train.bin")
test_graph_list, test_label_dict = load_graphs("./test.bin")

# TODO: implement Dataset subclass for loading train data
trainset = Dataset()

data_loader = DataLoader(trainset, batch_size=5, shuffle=True, collate_fn=collate)

test_bg = dgl.batch(test_graph_list)
test_y = torch.tensor(map(list, test_label_dict))

# Create model
NUM_PREDICTED_VALUES = 31
model = Regressor(1, 256, NUM_PREDICTED_VALUES)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()

# Train the model
EPOCHS = 10
epoch_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for iter_idx, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter_idx + 1)
    print(f'Epoch {epoch}, loss {epoch_loss}')
    epoch_losses.append(epoch_loss)

torch.save(model, "./gcn_model")

# Test the model
model.eval()
r2_scores = []
rmse_scores = []

for graph, true_y in zip(test_graph_list, test_label_dict):
    pred_y = model(graph)
    r2_score = metrics.r2_score(true_y, pred_y)
    r2_scores.append(r2_score)
    rmse_score = metrics.mean_squared_error(true_y, pred_y, squared=False)
    rmse_scores.append(rmse_score)

print(f'R2: {np.average(r2_scores)}, RMSE: {np.average(rmse_scores)}')
