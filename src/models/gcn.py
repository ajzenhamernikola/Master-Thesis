import os
from timeit import default_timer as timer
from time import localtime

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
from src.utils.FileLogger import FileLogger
from src.utils.IntProgress import IntProgress

DatasetClass = CNFDatasetNode2Vec
csv_file_x = os.path.join(os.path.dirname(__file__),
                          '..', '..', 'INSTANCES', 'chosen_data', 'splits.csv')
csv_file_y = os.path.join(os.path.dirname(__file__),
                          '..', '..', 'INSTANCES', 'chosen_data', 'all_data_y.csv')
root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'INSTANCES')

int_progress: IntProgress = None


def collate(dev):
    def collate_fn(samples):
        """
            Forms a mini-batch from a given list of graphs and label pairs
            :param samples: list of tuple pairs (graph, label)
            :return:
            """
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels, device=dev, dtype=torch.float32)
        if int_progress is not None:
            int_progress.step(batched_labels.shape[0])
        return batched_graph, batched_labels

    return collate_fn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation, activation_params, dropout_p,
                 pooling="avg"):
        super(Regressor, self).__init__()

        # Checks
        num_layers = len(hidden_layers)
        if num_layers < 1:
            raise ValueError(f"You must have at least one hidden layer. You passed {num_layers}: {hidden_layers}")

        if activation == "relu":
            self.activation = nn.ReLU(**activation_params)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(**activation_params)
        elif activation == "elu":
            self.activation = nn.ELU(**activation_params)
        elif activation == "selu":
            self.activation = nn.SELU(**activation_params)
        else:
            raise NotImplementedError(f"Unknown activation method: {pooling}")

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphConv(input_dim, hidden_layers[0]))
        self.bn_layers.append(nn.BatchNorm1d(hidden_layers[0]))
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_layers[i], hidden_layers[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
        self.layers.append(GraphConv(hidden_layers[-1], output_dim))

        # Additional layers
        if pooling == "avg":
            self.pool = AvgPooling()
        elif pooling == "sum":
            self.pool = SumPooling()
        elif pooling == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError(f"Unknown pooling method: {pooling}")

        self.linear = nn.Linear(hidden_layers[-1], output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, g: dgl.DGLGraph):
        # Use the already set features data as initial node features
        h = g.ndata['features']
        # Apply layers
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(g, h)
            if False: #i != len(self.layers) - 1:
                #h = self.bn_layers[i](h)
                h = self.activation(h)

        # Perform pooling over all nodes in each graph in every layer
        pooled_h = self.pool(g, h)

        # Return the predicted data in linear layer over pooled data
        # linear = self.linear(pooled_h)

        return pooled_h


def train_one_epoch(data_loader_train, loss_func, model, optimizer, train_device):
    model.train()
    train_loss = 0
    iter_idx = 0
    for iter_idx, (bg, label) in enumerate(data_loader_train):
        # Get the data
        inputs, labels = bg.to(train_device), label.to(train_device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)        
        loss.backward()
        optimizer.step()
        
        # Report loss
        train_loss += loss.detach().item()
    return np.round(train_loss / (iter_idx + 1), 3)


def validate_one_epoch(model, loss_func, data_loader_val, train_device, test_device):
    model.to(test_device)
    model.eval()
    val_loss = 0
    iter_num = 0
    y_pred = np.empty((0, 31))
    y_true = np.empty((0, 31))
    for iter_num, (bg, label) in enumerate(data_loader_val):
        prediction = model(bg.to(test_device))
        # Calculate loss
        loss = loss_func(prediction, label.to(test_device))
        val_loss += loss.detach().item()
        # Save prediction for calculating R^2 and RMSE scores
        pred_y = model(bg.to(test_device))
        pred_y = np.array(pred_y.to(test_device).detach().numpy())
        y_pred = np.vstack((y_pred, pred_y))
        y_true = np.vstack((y_true, label.to(test_device)))

    r2_score_val_avg, rmse_score_val_avg, _, _ = calculate_r2_and_rmse_scores(y_pred, y_true)
    model.to(train_device)

    return np.round([val_loss / (iter_num + 1), r2_score_val_avg, rmse_score_val_avg], 3)


def calculate_r2_and_rmse_scores(y_pred, y_true):
    r2_scores_val = np.empty((31,))
    rmse_scores_val = np.empty((31,))

    for i in range(31):
        r2_scores_val[i] = metrics.r2_score(y_true[:, i:i + 1], y_pred[:, i:i + 1])
        rmse_scores_val[i] = metrics.mean_squared_error(y_true[:, i:i + 1], y_pred[:, i:i + 1], squared=False)

    r2_score_val_avg = np.average(r2_scores_val)
    rmse_score_val_avg = np.average(rmse_scores_val)

    return r2_score_val_avg, rmse_score_val_avg, r2_scores_val, rmse_scores_val


def time_for_early_stopping(val_losses, no_progress_max):
    if len(val_losses) <= no_progress_max:
        return False

    last_epoch_loss = val_losses[-1]
    best_epoch_loss = np.min(val_losses)

    neg_slope = best_epoch_loss < last_epoch_loss
    no_significant_progress = best_epoch_loss - last_epoch_loss < 0.05 * np.average(val_losses[-no_progress_max:])

    return neg_slope or no_significant_progress


# Train the model
def train(train_device, test_device):
    global int_progress

    batch_size = 1

    # Load train data
    trainset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Train")
    data_loader_train = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate(train_device))

    # Load val data
    valset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Validation")
    data_loader_val = DataLoader(valset, batch_size=batch_size, shuffle=True, collate_fn=collate(train_device))

    # Load train+val data
    trainvalset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Train+Validation")
    data_loader_trainval = DataLoader(trainvalset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate(train_device))

    int_progress = IntProgress(0, trainvalset.__len__())

    print("\n")

    # Model params
    input_dim = trainset.hidden_features_dim
    output_dim = 31
    hidden_layers = [64, 64, 64]
    activation = "relu"
    activation_params = {}
    dropout_p = 0.0
    pooling = "avg"
    # Optimizer params
    lr = 1e-4
    w_decay = 0
    loss = "mse"
    # Num of epochs
    epochs = 200
    print("Enter the number of epochs required for training before checking for early stopping")
    no_progress_max = int(input())

    # Checks
    if loss == "l1":
        loss_func = nn.L1Loss()
    elif loss == "mse":
        loss_func = nn.MSELoss()
    elif loss == "smooth-l1":
        loss_func = nn.SmoothL1Loss()
    elif loss == "rmse":
        loss_func = RMSELoss()
    else:
        raise ValueError(f"Loss function is not one of ['l1', 'mse', 'smooth-l1']. You passed: {loss}")

    # Model name
    hidden_layers_str = '-'.join(list(map(lambda x: str(x), hidden_layers)))
    activation_params_str = ''
    for key in activation_params.keys():
        activation_params_str += key + "+" + str(activation_params[key]) + ','

    mfn = f"gnn_{input_dim}_{output_dim}_" + \
          f"{hidden_layers_str}_{activation}_{activation_params_str}_" + \
          f"{dropout_p}_{pooling}_{lr}_{w_decay}_{loss}_{epochs}_{no_progress_max}"
    model_root = os.path.join(os.path.dirname(__file__), '..', '..', 'models', f"model_{mfn}")
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # Create a logger
    log = FileLogger(model_root, mfn)

    log.log_bar()
    log.log_line(f"Model name: {mfn}")
    log.log_bar()
    log.log_line("Model parameters")
    log.log_bar()
    log.log_line(f"Features dimension: {input_dim}")
    log.log_line(f"Output dimension: {output_dim}")
    log.log_line(f"Graph convolution hidden layers: {hidden_layers}")
    log.log_line(f"Activation function: {activation}")
    log.log_line(f"Activation function params: {activation_params}")
    log.log_line(f"Dropout: {dropout_p}")
    log.log_line(f"Pooling method: {pooling}")
    log.log_bar()
    log.log_line("Optimizer params (ADAM)")
    log.log_bar()
    log.log_line(f"Learning rate: {lr}")
    log.log_line(f"Weight decay: {w_decay}")
    log.log_line(f"Loss function: {loss}")
    log.log_bar()
    log.log_line(f"Maximum number of epochs: {epochs}")
    log.log_line(f"Number of epochs checked for early stopping: {no_progress_max}")
    log.log_bar()

    # Choose whether to overwrite existing model, train the existing model more or create a new model
    overwrite = True
    model_path = os.path.join(model_root, mfn)
    if os.path.exists(model_path):
        print("\nModel already exist. Would you like to overwrite it [o], continue training [t], or use existing [e]?")
        response = input().lower()
        if response == "e":
            return model_path, log
        if response == "t":
            overwrite = False
        elif response != "o":
            raise ValueError(f"You responded with \"{response}\", while the options are one of {{o, t, e}}")

    # Create model
    if overwrite:
        model = Regressor(input_dim,
                          output_dim,
                          hidden_layers,
                          activation,
                          activation_params,
                          dropout_p,
                          pooling)
        train_losses = []
        val_losses = []
        r2_scores = []
        rmse_scores = []
        train_times = []
        val_times = []
        best_epoch = -1
        best_val_loss = None
        current_epoch = 0
    else:
        data = torch.load(model_path)
        model = data[0]
        train_losses = data[1]
        val_losses = data[2]
        r2_scores = data[3]
        rmse_scores = data[4]
        train_times = data[5]
        val_times = data[6]
        best_epoch = np.argmin(val_losses)
        best_val_loss = np.min(val_losses)
        current_epoch = len(train_losses)
        print("How much to train before checking for early stopping?")
        no_progress_max = current_epoch + int(input())
        
    model.to(train_device)

    log.log_line("\nModel created! Training...\n")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

    # Start training
    log.log_bar()
    t = localtime()
    log.log_line(f"Started training at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}.{t.tm_year}.")
    log.log_bar()

    while True:
        current_epoch += 1
        if current_epoch == epochs:
            break

        t = localtime()
        log.log_line(
            f"\nStarted epoch {current_epoch} at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}." +
            f"{t.tm_year}.")
        int_progress.reset()

        # Train on training data
        time_start = timer()
        train_loss = train_one_epoch(data_loader_train, loss_func, model, optimizer, train_device)
        time_elapsed = timer() - time_start
        train_times.append(time_elapsed)
        train_losses.append(train_loss)

        # Validate on validating data
        time_start = timer()
        val_loss, r2_score_val_avg, rmse_score_val_avg = validate_one_epoch(model, loss_func, data_loader_val,
                                                                            train_device, test_device)
        time_elapsed = timer() - time_start
        val_times.append(time_elapsed)
        val_losses.append(val_loss)
        r2_scores.append(r2_score_val_avg)
        rmse_scores.append(rmse_score_val_avg)

        # Remember the best validation loss
        if best_val_loss is None:
            best_val_loss = val_loss
            best_epoch = current_epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = current_epoch

        log.log_line(f'\nEpoch {current_epoch} summary:')
        log.log_line(f'\tTrain loss: {train_loss}; training time: {train_times[-1]:.2f}s')
        log.log_line(f'\tValidation loss: {val_loss}; validation time: {val_times[-1]:.2f}s')
        log.log_line(f'\tValidation R^2 score: {r2_score_val_avg}')
        log.log_line(f'\tValidation RMSE score: {rmse_score_val_avg}')
        
        # Serialize model for later usage
        torch.save([model, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, []], model_path)

        # Early stopping
        if time_for_early_stopping(val_losses, no_progress_max):
            print("Early stopping flag occured. Would you like to continue training? [y/n]")
            continue_train = input() == "y"
            if not continue_train:
                log.log_line(
                    f'\nThe training is stopped in epoch {current_epoch} due to no progress in validation loss:')
                log.log_line(f'\tBest validation loss {best_val_loss} achieved in epoch {best_epoch}')
                break

            print(f"How many epochs to train from epoch {no_progress_max} until next early stopping?")
            no_progress_max += int(input())

    # Plot graphs
    best_val_loss = np.argmin(val_losses)
    highest_val_loss = np.maximum(np.max(val_losses), np.max(train_losses))

    fig, (top_ax, bot_ax) = plt.subplots(2)
    fig.suptitle("Training progress")
    fig.set_size_inches(w=len(train_losses) * 0.75, h=15)

    top_ax.set_ylabel("Loss value")
    top_ax.plot(range(len(train_losses)), train_losses, color="blue", linestyle="solid", label="train loss")
    top_ax.plot(range(len(val_losses)), val_losses, color="red", linestyle="solid", label="val loss")
    top_ax.plot([best_val_loss, best_val_loss], [0, highest_val_loss], color="green", linestyle="solid",
                label="best epoch")

    bot_ax.set_ylabel("Score value")
    bot_ax.plot(range(len(r2_scores)), r2_scores, color="magenta", linestyle="solid", label="val R^2 score")
    bot_ax.plot(range(len(rmse_scores)), rmse_scores, color="orange", linestyle="solid", label="val RMSE score")
    bot_ax.legend()

    for ax in fig.get_axes():
        ax.set_xticks(range(len(train_losses)))
        ax.set_xticklabels(range(1, len(train_losses) + 1))
        ax.set_xlabel("Epoch #")
        ax.legend()

    plt.savefig(model_path + '.png')
    plt.close()

    # Serialize model for later usage
    torch.save([model, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, []], model_path)

    need_to_retrain = input("\nShould I retrain the model? [y/n] ") == "y"
    if need_to_retrain:
        # Retraining model on train+val dataset
        log.log_bar()
        t = localtime()
        log.log_line(f"Started retraining at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}.{t.tm_year}.")
        log.log_bar()

        log.log_line("\nRetraining model on train+val dataset...")
        retrain_times = []

        for current_epoch in range(1, best_epoch + 1):
            t = localtime()
            log.log_line(
                f"\tStarted epoch {current_epoch} at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}." +
                f"{t.tm_year}.")
            int_progress.reset()
    
            time_start = timer()
            train_one_epoch(data_loader_trainval, loss_func, model, optimizer, train_device)
            time_elapsed = timer() - time_start
            retrain_times.append(time_elapsed)
            log.log_line(f"\tFinished epoch {current_epoch} of {best_epoch}")
    
        # Serialize model for later usage
        torch.save([model, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, retrain_times],
                   model_path)
        
    return model_path, log


# Test the model
def test(predict_device, test_device, model_path: str, log: FileLogger):
    global int_progress

    testset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Test")
    data_loader_test = DataLoader(testset, batch_size=1, shuffle=True, collate_fn=collate(predict_device))

    int_progress = IntProgress(0, testset.__len__())

    # Load the model
    data = torch.load(model_path)
    model = data[0]
    model.to(predict_device)

    # Prepare for predicting
    log.log_bar()
    t = localtime()
    log.log_line(f"Started testing at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}.{t.tm_year}.")
    log.log_bar()

    model.eval()
    y_pred = np.empty((0, 31))
    y_true = np.empty((0, 31))

    # Predict
    log.log_line("\nPredicting...\n")
    pred_data_name = model_path + '.predicted'
    true_data_name = model_path + '.true'
    # Check if there are predicted data for this model, and load it
    pred_exists = os.path.exists(pred_data_name + '.npy')
    true_exists = os.path.exists(true_data_name + '.npy')
    if pred_exists and true_exists:
        y_pred = np.load(pred_data_name + '.npy')
        y_true = np.load(true_data_name + '.npy')
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
    log.log_line("\nEvaluating...")
    r2_score_test_avg, rmse_score_test_avg, r2_scores_test, rmse_scores_test = \
        calculate_r2_and_rmse_scores(y_pred, y_true)
    log.log_line(f'\nAverage R2 score: {r2_score_test_avg:.4f}')
    log.log_line(f'Average RMSE score: {rmse_score_test_avg:.4f}\n')

    # Prediction graphs per solver
    png_file = f"{model_path}_scores_per_solver.png"

    solver_names = ["ebglucose", "ebminisat", "glucose2", "glueminisat", "lingeling", "lrglshr", "minisatpsm",
                    "mphaseSAT64", "precosat", "qutersat", "rcl", "restartsat", "cryptominisat2011", "spear-sw",
                    "spear-hw", "eagleup", "sparrow", "marchrw", "mphaseSATm", "satime11", "tnm", "mxc09", "gnoveltyp2",
                    "sattime", "sattimep", "clasp2", "clasp1", "picosat", "mphaseSAT", "sapperlot", "sol"]
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    xticks = range(1, len(r2_scores_test) + 1)
    ymin = int(np.floor(np.min(r2_scores_test)))
    yticks = np.linspace(ymin, 1, 10*(1-ymin))
    ylabels = np.round(yticks, 1)
    plt.title("R2 scores per solver")
    plt.xticks(ticks=xticks, labels=list(solver_names), rotation=90)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.ylim((np.min(yticks), np.max(yticks)))
    plt.bar(xticks, r2_scores_test, color="#578FF7")
    plt.plot([xticks[0], xticks[-1]], [r2_score_test_avg, r2_score_test_avg], "r-")

    plt.subplot(1, 2, 2)
    xticks = range(1, len(rmse_scores_test) + 1)
    rmse_score_test_max = np.ceil(np.max(rmse_scores_test))
    yticks = np.linspace(0, rmse_score_test_max, 10)
    ylabels = np.round(np.linspace(0, rmse_score_test_max, 10), 1)
    plt.title("RMSE scores per solver")
    plt.xticks(ticks=xticks, labels=list(solver_names), rotation=90)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.ylim((np.min(yticks), np.max(yticks)))
    plt.bar(xticks, rmse_scores_test, color="#FA6A68")
    plt.plot([xticks[0], xticks[-1]], [rmse_score_test_avg, rmse_score_test_avg], "b-")

    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    plt.close()

    # More details
    train_times = data[5]
    val_times = data[6]
    retrain_times = data[7]

    log.log_bar()
    log.log_line("Average data")
    log.log_bar()
    log.log_line(f"Training time: {np.average(train_times):.2f}s")
    log.log_line(f"Validating time: {np.average(val_times):.2f}s")
    log.log_line(f"Retraining time: {np.average(retrain_times):.2f}s")

    log.log_bar()
    log.log_line("Total data")
    log.log_bar()
    log.log_line(f"Training time: {np.sum(train_times):.2f}s")
    log.log_line(f"Validating time: {np.sum(val_times):.2f}s")
    log.log_line(f"Retraining time: {np.sum(retrain_times):.2f}s")
