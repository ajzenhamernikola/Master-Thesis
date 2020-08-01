import os
import pickle as pkl
import gc

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from .classes import Predictor


def load_next_batch(cnf_dir: str, instance_ids: list, selected_idx: list, splits: dict, dataset_type: str):
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
        pickle_file = os.path.join(cnf_dir, instance_id + ".dgcnn.pickled")
        with open(pickle_file, "rb") as f:
            batch_graph.append(pkl.load(f))
            labels.append(batch_graph[-1].labels)

    return batch_graph, labels


def loop_dataset(cnf_dir: str, model_output_dir: str, model: str, instance_ids: list, splits: dict, epoch: int,
                 classifier: Predictor, sample_idxes: list, optimizer=None, batch_size=1, dataset_type="Train",
                 print_auc=False):
    total_loss = []
    total_iters = (len(sample_idxes) + (batch_size - 1) * (optimizer is None)) // batch_size
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * batch_size: (pos + 1) * batch_size]

        batch_graph, targets = load_next_batch(cnf_dir, instance_ids, selected_idx, splits, dataset_type)
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
        
        del batch_graph
        
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
        gc.collect()
        
    if optimizer is None:
        assert n_samples == len(sample_idxes)
        
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    if dataset_type == "Test":
        predictions_filename = os.path.join(model_output_dir, model, "test_ypred.txt")
    else:
        predictions_filename = os.path.join(model_output_dir, model, f"{dataset_type}_{epoch}_outputs.txt")
    np.savetxt(predictions_filename, all_scores)  # output predictions

    if not classifier.regression and print_auc:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0]))
        
    del all_targets
    gc.collect()

    return avg_loss
