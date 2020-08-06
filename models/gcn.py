import os
from timeit import default_timer as timer
from time import localtime
import gc

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import GraphConv, SumPooling, MaxPooling, AvgPooling
from sklearn import metrics
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

from .common.CNFDatasetNode2Vec import CNFDatasetNode2Vec


DatasetClass = CNFDatasetNode2Vec
csv_file_x = os.path.join('.', 'INSTANCES', 'splits.csv')
csv_file_y = os.path.join('.', 'INSTANCES', 'all_data_y.csv')
root_dir = os.path.join('.', 'INSTANCES',)


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
        # Input layer
        self.layers.append(GraphConv(input_dim, hidden_layers[0]))
        self.bn_layer = nn.BatchNorm1d(hidden_layers[0])
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_layers[i], hidden_layers[i + 1]))

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
            if i != 0:
                h = self.dropout(h)
            else:
                h = self.bn_layer(h)
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)

        # Perform pooling over all nodes in each graph in every layer
        pooled_h = self.pool(g, h)

        # Return the predicted data in linear layer over pooled data
        linear = self.linear(pooled_h)

        return linear


def train_one_epoch(data_loader_train, num_of_batches, loss_func, model, optimizer, train_device):
    model.train()
    train_loss = 0
    iter_idx = 0
    pbar = tqdm(total=num_of_batches, unit='batch')

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
        curr_loss = loss.detach().item()
        train_loss += curr_loss
        pbar.update(n=1)
        pbar.set_description(f'Train loss: {curr_loss:.5f}')
        
        del inputs
        del labels
        gc.collect()

    pbar.close()

    return np.round(train_loss / (iter_idx + 1), 3)


def validate_one_epoch(model_root, epoch, predictor, loss_func, data_loader_val, num_of_batches, train_device,
                       test_device):
    predictor.to(test_device)
    predictor.eval()
    val_loss = 0
    iter_num = 0
    y_pred = np.empty((0, 31))
    y_true = np.empty((0, 31))
    pbar = tqdm(total=num_of_batches, unit='batch')

    for iter_num, (bg, label) in enumerate(data_loader_val):
        # Get the data
        prediction = predictor(bg.to(test_device))
        
        # Calculate loss
        loss = loss_func(prediction, label.to(test_device))
        curr_loss = loss.detach().item()
        val_loss += curr_loss
        
        # Save prediction for calculating R^2 and RMSE scores
        pred_y = predictor(bg.to(test_device))
        pred_y = np.array(pred_y.to(test_device).detach().numpy())
        y_pred = np.vstack((y_pred, pred_y))
        y_true = np.vstack((y_true, label.to(test_device)))

        pbar.update(n=1)
        pbar.set_description(f'Validation loss: {curr_loss:.5f}')

    pbar.close()

    outputs_filename = os.path.join(model_root, f"Validation_{epoch}_outputs.txt")
    np.savetxt(outputs_filename, y_pred, "%.6f")
    ytrue_filename = os.path.join(model_root, f"Validation_y_true.txt")
    if not os.path.exists(ytrue_filename):
        np.savetxt(ytrue_filename, y_true, "%.6f")

    r2_score_val_avg, rmse_score_val_avg, _, _ = calculate_r2_and_rmse_scores(y_pred, y_true)
    predictor.to(train_device)

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
def train(model_output_dir, model, train_device, test_device):
    # Load train data
    batch_size = 50
    trainset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Train")
    data_loader_train = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate(train_device))
    train_num_of_batches = int(np.ceil(trainset.__len__() / batch_size))

    # Load val data
    val_batch_size = 1
    valset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Validation")
    data_loader_val = DataLoader(valset, batch_size=val_batch_size, shuffle=False, collate_fn=collate(train_device))
    val_num_of_batches = int(np.ceil(valset.__len__() / val_batch_size))

    # Load train+val data
    retrain_batch_size = 50
    trainvalset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Train+Validation")
    data_loader_trainval = DataLoader(trainvalset, batch_size=retrain_batch_size, shuffle=True,
                                      collate_fn=collate(train_device))
    retrain_num_of_batches = int(np.ceil(valset.__len__() / retrain_batch_size))

    testset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Test")

    print("\n")

    # Model params
    input_dim = trainset.hidden_features_dim
    output_dim = 31
    hidden_layers = [64, 64, 64]
    activation = "leaky"
    activation_params = {"negative_slope": 0.1}
    dropout_p = 0.0
    pooling = "avg"
    # Optimizer params
    lr = 1e-3
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
    model_root = os.path.join(model_output_dir, model)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    print(80 * "=")
    print(f"Initializing {model}")
    print(80 * "=")
    print("Model parameters")
    print(80 * "=")
    print(f"Features dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Graph convolution hidden layers: {hidden_layers}")
    print(f"Activation function: {activation}")
    print(f"Activation function params: {activation_params}")
    print(f"Dropout: {dropout_p}")
    print(f"Pooling method: {pooling}")
    print(80 * "=")
    print("Optimizer params (ADAM)")
    print(80 * "=")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {w_decay}")
    print(f"Loss function: {loss}")
    print(80 * "=")
    print(f"Maximum number of epochs: {epochs}")
    print(f"Number of epochs checked for early stopping: {no_progress_max}")
    print(80 * "=")

    # Choose whether to overwrite existing model, train the existing model more or create a new model
    overwrite = True
    retrain = False
    model_path = os.path.join(model_root, "best_GCN_model")
    if os.path.exists(model_path):
        print("\nModel already exist. Would you like to overwrite it [o], continue training [t], just retrain [r] " + 
              "or use existing [e]?")
        response = input().lower()
        if response == "e":
            return model_path
        if response == "t":
            overwrite = False
        if response == "r":
            overwrite = False
            retrain = True
        elif response != "o":
            raise ValueError(f"You responded with \"{response}\", while the options are one of {{o, t, e, r}}")

    # Create model
    if overwrite:
        predictor = Regressor(input_dim,
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
        predictor = data[0]
        train_losses = data[1]
        val_losses = data[2]
        r2_scores = data[3]
        rmse_scores = data[4]
        train_times = data[5]
        val_times = data[6]
        losses = np.array(np.array(train_losses) + np.array(val_losses)) / 2
        best_epoch = np.argmin(losses) # np.argmin(val_losses)
        best_val_loss = val_losses[best_epoch]
        current_epoch = len(train_losses)
        if not retrain:
            print("How much to train before checking for early stopping?")
            no_progress_max = current_epoch + int(input())
        
    predictor.to(train_device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=w_decay)

    if not retrain:
        print("\nModel created! Training...\n")

        # Start training
        print(80 * "=")
        t = localtime()
        print(f"Started training at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}.{t.tm_year}.")
        print(80 * "=")

        while True:
            current_epoch += 1
            if current_epoch == epochs:
                break

            t = localtime()
            print(
                f"Training epoch {current_epoch} started at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}." +
                f"{t.tm_year}.")

            # Train on training data
            time_start = timer()
            train_loss = train_one_epoch(data_loader_train, train_num_of_batches, loss_func, predictor, optimizer, train_device)
            time_elapsed = timer() - time_start
            train_times.append(time_elapsed)
            train_losses.append(train_loss)

            print(
                f"Validating epoch {current_epoch} started at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}." +
                f"{t.tm_year}.")

            # Validate on validating data
            time_start = timer()
            val_loss, r2_score_val_avg, rmse_score_val_avg = validate_one_epoch(model_root, current_epoch, predictor, loss_func, data_loader_val,
                                                                                val_num_of_batches, train_device, test_device)
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

            print(f'\nEpoch {current_epoch} summary:')
            print(f'\tTrain loss: {train_loss}; training time: {train_times[-1]:.2f}s')
            print(f'\tValidation loss: {val_loss}; validation time: {val_times[-1]:.2f}s')
            print(f'\tValidation R^2 score: {r2_score_val_avg}')
            print(f'\tValidation RMSE score: {rmse_score_val_avg}')
            
            # Serialize model for later usage
            torch.save([predictor, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, []], model_path)

            # Early stopping
            if time_for_early_stopping(val_losses, no_progress_max):
                print("Early stopping flag occured. Would you like to continue training? [y/n]")
                continue_train = input() == "y"
                if not continue_train:
                    print(
                        f'\nThe training is stopped in epoch {current_epoch} due to no progress in validation loss:')
                    print(f'\tBest validation loss {best_val_loss} achieved in epoch {best_epoch}')
                    break

                print(f"How many epochs to train from epoch {no_progress_max} until next early stopping?")
                no_progress_max += int(input())

        # Serialize model for later usage
        torch.save([predictor, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, []], model_path)
        
        print("Do you want to retrain the model? [y/n]")
        retrain = input() == "y"
    
    if retrain:
        print(f"Best epoch is {best_epoch}. Do you want to change the number of epochs for retraining? [y/n]")
        need_to_change = input() == "y"
        if need_to_change:
            print("Enter the number of epochs for retraining the model:")
            best_epoch = int(input())
        
        # Retraining model on train+val dataset
        print(80 * "=")
        t = localtime()
        print(f"Started retraining at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}.{t.tm_year}.")
        print(80 * "=")

        print("\nRetraining model on train+val dataset...")
        retrain_times = []

        for current_epoch in range(1, best_epoch + 1):
            t = localtime()
            print(
                f"\tStarted epoch {current_epoch} at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}." +
                f"{t.tm_year}.")
    
            time_start = timer()
            train_one_epoch(data_loader_trainval, retrain_num_of_batches, loss_func, predictor, optimizer, train_device)
            time_elapsed = timer() - time_start
            retrain_times.append(time_elapsed)
            print(f"\tFinished epoch {current_epoch} of {best_epoch}")
    
        # Serialize model for later usage
        torch.save([predictor, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, retrain_times],
                   model_path)
        
    return model_path


# Test the model
def test(model_output, model, predict_device, test_device):
    test_batch_size = 1
    testset = DatasetClass(csv_file_x, csv_file_y, root_dir, "Test")
    data_loader_test = DataLoader(testset, batch_size=test_batch_size, shuffle=False, collate_fn=collate(predict_device))
    test_num_of_batches = int(np.ceil(testset.__len__() / test_batch_size))

    # Load the model
    data = torch.load(os.path.join(model_output, model, "best_GCN_model"))
    predictor = data[0]
    predictor.to(predict_device)

    # Prepare for predicting
    print(80 * "=")
    t = localtime()
    print(f"Started testing at: {t.tm_hour}:{t.tm_min}:{t.tm_sec} {t.tm_mday}.{t.tm_mon}.{t.tm_year}.")
    print(80 * "=")

    predictor.eval()
    y_pred = np.empty((0, 31))
    y_true = np.empty((0, 31))

    # Predict
    print("\nPredicting...\n")
    
    pred_data_name = os.path.join(model_output, model, "Test_ypred.txt")
    true_data_name = os.path.join(model_output, model, "Test_ytrue.txt")
    if os.path.exists(pred_data_name) and os.path.exists(true_data_name):
        y_pred = np.loadtxt(pred_data_name)
        y_true = np.loadtxt(true_data_name)
    else:
        with torch.no_grad():
            pbar = tqdm(total=test_num_of_batches, unit='batch')
            
            for i, (graph, true_y) in enumerate(data_loader_test):
                pred_y = predictor(graph.to(predict_device))
                pred_y = np.array(pred_y.to(test_device))
                y_pred = np.vstack((y_pred, pred_y))
                y_true = np.vstack((y_true, true_y.to(test_device)))
                
                pbar.update(n=1)
                pbar.set_description(f'Testing the model...')

            pbar.close()
            # Save the predicted data
            np.savetxt(pred_data_name, y_pred, "%.6f")
            np.savetxt(true_data_name, y_true, "%.6f")

    # Evaluate
    print("\nEvaluating...")
    r2_score_test_avg, rmse_score_test_avg, r2_scores_test, rmse_scores_test = \
        calculate_r2_and_rmse_scores(y_pred, y_true)
    print(f'\nAverage R2 score: {r2_score_test_avg:.4f}')
    print(f'Average RMSE score: {rmse_score_test_avg:.4f}\n')

    # Prediction graphs per solver
    png_file = os.path.join(model_output, model, "GCN.png")

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
    
    # Plot epochs
    train_losses = data[1]
    val_losses = data[2]
    r2_scores = data[3]
    rmse_scores = data[4]
    
    best_val_loss = np.argmin(val_losses)
    highest_val_loss = np.maximum(np.max(val_losses), np.max(train_losses))

    fig, (top_ax, bot_ax) = plt.subplots(2)
    plt.title("Training progress")
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

    plt.savefig(os.path.join(model_output, model, "GCN_losses.png"))
    plt.close()

    # More details
    train_times = data[5]
    val_times = data[6]
    retrain_times = data[7]

    print(80 * "=")
    print("Average data")
    print(80 * "=")
    print(f"Training time: {np.average(train_times):.2f}s")
    print(f"Validating time: {np.average(val_times):.2f}s")
    print(f"Retraining time: {0 if len(retrain_times) == 0 else np.average(retrain_times):.2f}s")

    print(80 * "=")
    print("Total data")
    print(80 * "=")
    print(f"Training time: {np.sum(train_times):.2f}s")
    print(f"Validating time: {np.sum(val_times):.2f}s")
    print(f"Retraining time: {np.sum(retrain_times):.2f}s")
