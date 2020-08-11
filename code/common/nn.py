import os
import gc

import numpy as np
from tqdm import tqdm
import dgl
import torch

from .process_results import calculate_r2_and_rmse_metrics


def collate(dev):
    def collate_fn(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels, device=dev, dtype=torch.float32)
        return batched_graph, batched_labels

    return collate_fn


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

    r2_score_val_avg, rmse_score_val_avg, _, _ = calculate_r2_and_rmse_metrics(None, None, y_true, y_pred)
    predictor.to(train_device)

    return np.round([val_loss / (iter_num + 1), r2_score_val_avg, rmse_score_val_avg], 3)


def time_for_early_stopping(val_losses: list, look_behind: int):
    if len(val_losses) < look_behind:
        return False

    last_epoch_loss = val_losses[-1]
    avg_epoch_loss = np.average(val_losses[-look_behind:])

    # Stop training if the progress in last epoch is less than 5% of average losses
    return avg_epoch_loss - last_epoch_loss < 0.05 * avg_epoch_loss
