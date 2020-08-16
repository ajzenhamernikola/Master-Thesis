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
from tqdm import tqdm

from .common.nn import collate, train_one_epoch, validate_one_epoch, time_for_early_stopping
from .common.process_results import calculate_r2_and_rmse_metrics_nn, plot_r2_and_rmse_scores_nn


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation, activation_params, dropout_p,
                 pooling="avg"):
        super(GCN, self).__init__()

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
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Unknown activation method: {pooling}")

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphConv(input_dim, hidden_layers[0]))
        self.bn_layer = nn.BatchNorm1d(input_dim)
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
            h = self.activation(h)

        # Perform pooling over all nodes in each graph in every layer
        pooled_h = self.pool(g, h)

        # Return the predicted data in linear layer over pooled data
        linear = self.linear(pooled_h)

        return linear


# Train the model
def train(model_output_dir, model, trainset, valset, trainvalset, train_device, test_device):
    # Load train data
    batch_size = 40
    data_loader_train = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate(train_device))
    train_num_of_batches = int(np.ceil(trainset.__len__() / batch_size))

    # Load val data
    val_batch_size = 1
    data_loader_val = DataLoader(valset, batch_size=val_batch_size, shuffle=False, collate_fn=collate(train_device))
    val_num_of_batches = int(np.ceil(valset.__len__() / val_batch_size))

    # Load train+val data
    retrain_batch_size = batch_size
    data_loader_trainval = DataLoader(trainvalset, batch_size=retrain_batch_size, shuffle=True,
                                      collate_fn=collate(train_device))
    retrain_num_of_batches = int(np.ceil(trainvalset.__len__() / retrain_batch_size))

    print("\n")

    # Model params
    input_dim = trainset.hidden_features_dim
    output_dim = 31
    hidden_layers = [64, 64, 64]
    activation = "leaky"
    activation_params = {"negative_slope": 0.2}
    dropout_p = 0.0
    pooling = "avg"
    # Optimizer params
    lr = 0.001
    w_decay = 0
    loss = "mse"
    # Num of epochs
    epochs = 200
    no_progress_max = 50

    # Checks
    if loss == "l1":
        loss_func = nn.L1Loss()
    elif loss == "mse":
        loss_func = nn.MSELoss()
    elif loss == "smooth-l1":
        loss_func = nn.SmoothL1Loss()
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
    model_exists = os.path.exists(model_path)
    if model_exists:
        print("\nModel already exist. Would you like to overwrite it [o], continue training [t], just retrain [r] " +
              "or use existing [e]?")
        response = input().lower()
        if response == "e":
            return model_path
        elif response == "t":
            overwrite = False
        elif response == "r":
            overwrite = False
            retrain = True
        elif response != "o":
            raise ValueError(f"You responded with \"{response}\", while the options are one of {{o, t, e, r}}")
    else:
        predictor = GCN(input_dim,
                        output_dim,
                        hidden_layers,
                        activation,
                        activation_params,
                        dropout_p,
                        pooling)

    # Create model
    if overwrite:
        train_losses = []
        val_losses = []
        r2_scores = []
        rmse_scores = []
        train_times = []
        val_times = []
        retrain_times = []
        best_epoch = -1
        best_val_loss = None
        current_epoch = 0
        print("Enter the number of epochs required for training before checking for early stopping")
        no_progress_max = int(input())
    else:
        data = torch.load(model_path)
        predictor = data[0]
        train_losses = data[1]
        val_losses = data[2]
        r2_scores = data[3]
        rmse_scores = data[4]
        train_times = data[5]
        val_times = data[6]
        retrain_times = data[7]
        best_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_epoch]
        current_epoch = len(train_losses)
        if not retrain:
            print(f"How much to train from {current_epoch} before checking for early stopping?")
            no_progress_max = current_epoch + int(input())

    if not model_exists or overwrite or (retrain and len(retrain_times) == 0):
        predictor = GCN(input_dim,
                        output_dim,
                        hidden_layers,
                        activation,
                        activation_params,
                        dropout_p,
                        pooling)
                        
    predictor.to(train_device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=w_decay)

    print(predictor)

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
            print(f'\tValidation RMSE score: {rmse_score_val_avg}\n')
            
            # Serialize model for later usage
            torch.save([predictor, train_losses, val_losses, r2_scores, rmse_scores, train_times, val_times, []], model_path)

            # Early stopping
            if time_for_early_stopping(val_losses, no_progress_max):
                print("Early stopping flag occured. Would you like to continue training? [y/n]")
                continue_train = input() == "y"
                if not continue_train:
                    print(
                        f'\nThe training is stopped in epoch {current_epoch} due to no progress in validation loss:')
                    print(f'\tBest validation loss {best_val_loss} achieved in epoch {best_epoch}\n')
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

        current_epoch = len(retrain_times) + 1
        while True:
            if current_epoch == best_epoch + 1:
                print("Do you want to retrain more? [y/n]")
                more_retraining = input() == "y"
                if not more_retraining:
                    break
                
                print(f"Enter the number of epochs to train from {best_epoch+1}:")
                best_epoch += int(input())
            
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
                       
            current_epoch += 1
        
    return model_path


# Test the model
def test(model_output, model, testset, predict_device, test_device):
    test_batch_size = 1
    data_loader_test = DataLoader(testset, batch_size=test_batch_size, shuffle=False, collate_fn=collate(predict_device))
    test_num_of_batches = int(np.ceil(testset.__len__() / test_batch_size))

    # Load the model
    model_output_dir = os.path.join(model_output, model)
    data = torch.load(os.path.join(model_output_dir, f"best_{model}_model"))
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
                pbar.set_description(f'Testing loss: {metrics.mean_squared_error(y_true, y_pred)}')

            pbar.close()
            # Save the predicted data
            np.savetxt(pred_data_name, y_pred, "%.6f")
            np.savetxt(true_data_name, y_true, "%.6f")

    # Evaluate
    print("\nEvaluating...")
    _, _, r2_scores_test, rmse_scores_test = \
        calculate_r2_and_rmse_metrics_nn(predictor, model_output_dir, model)

    plot_r2_and_rmse_scores_nn(r2_scores_test, rmse_scores_test, model_output_dir, model)
    
    # Plot epochs
    train_losses = np.clip(data[1], 0, 5)
    val_losses = np.clip(data[2], 0, 5)
    
    best_val_loss = np.argmin(val_losses)
    highest_val_loss = np.maximum(np.max(val_losses), np.max(train_losses))

    plt.title("Training progress")
    plt.figure(figsize=(6, 5))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss value")
    xticks = range(len(train_losses))
    plt.plot(xticks, train_losses, color="blue", linestyle="solid", label="train loss")
    plt.plot(xticks, val_losses, color="red", linestyle="solid", label="val loss")
    plt.plot([best_val_loss, best_val_loss], [0, highest_val_loss], color="green", linestyle="solid", label="best epoch")
    plt.ylim(0, 5)
    plt.legend()

    plt.tight_layout()
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
