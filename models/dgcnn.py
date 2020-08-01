import random
import os
import gc

import numpy as np
import pandas as pd
import torch

from .libs.dgcnn.classes import DGCNNPredictor
from .libs.dgcnn.util import loop_dataset, load_next_batch
from .common.nn import time_for_early_stopping


def train_one_epoch(dgcnn: DGCNNPredictor, epoch: int, batch_size: int, idxes: list, train_losses: dict,
                    dataset_type: str, print_auc: bool):
    random.shuffle(idxes)
    dgcnn.predictor.train()
    avg_loss = loop_dataset(cnf_dir=dgcnn.cnf_dir,
                            model_output_dir=dgcnn.model_output_dir,
                            model=dgcnn.model,
                            instance_ids=dgcnn.instance_ids,
                            splits=dgcnn.splits,
                            epoch=epoch,
                            classifier=dgcnn.predictor,
                            sample_idxes=idxes,
                            optimizer=dgcnn.optimizer,
                            batch_size=batch_size,
                            dataset_type=dataset_type)
    if not print_auc:
        avg_loss[2] = 0.0
    if dgcnn.regression:
        print('\033[92m  Average training of epoch %d: loss %.5f mae %.5f\033[0m' % (
            epoch, avg_loss[0], avg_loss[1]))
        train_losses["mse"].append(avg_loss[0])
        train_losses["mae"].append(avg_loss[1])
    else:
        print('\033[92m  Average training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, avg_loss[0], avg_loss[1], avg_loss[2]))


def validate_one_epoch(dgcnn: DGCNNPredictor, epoch: int, batch_size: int, val_losses: dict, print_auc: bool):
    dgcnn.predictor.eval()
    val_loss = loop_dataset(cnf_dir=dgcnn.cnf_dir,
                            model_output_dir=dgcnn.model_output_dir,
                            model=dgcnn.model,
                            instance_ids=dgcnn.instance_ids,
                            splits=dgcnn.splits,
                            epoch=epoch,
                            classifier=dgcnn.predictor,
                            sample_idxes=list(range(dgcnn.splits["Validation"])),
                            optimizer=None,
                            batch_size=batch_size,
                            dataset_type="Validation")
    if not print_auc:
        val_loss[2] = 0.0
    if dgcnn.regression:
        print('\033[92m  Average validation of epoch %d: loss %.5f mae %.5f\033[0m' % (
            epoch, val_loss[0], val_loss[1]))
        val_losses["mse"].append(val_loss[0])
        val_losses["mae"].append(val_loss[1])
    else:
        print('\033[92m  Average validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, val_loss[0], val_loss[1], val_loss[2]))


def train(dgcnn: DGCNNPredictor, num_epochs: int, batch_size: int, look_behind: int,
          print_auc=False, extract_features=False):
    train_idxes = list(range(dgcnn.splits["Train"]))

    best_loss = None
    best_epoch = None
    train_losses = {"mse": [], "mae": []}
    val_losses = {"mse": [], "mae": []}

    loss_for_early_stopping = "mse"
    for epoch in range(num_epochs):
        # Train one epoch
        train_one_epoch(dgcnn, epoch, batch_size, train_idxes, train_losses, "Train", print_auc)

        # Validate one epoch
        validate_one_epoch(dgcnn, epoch, batch_size, val_losses, print_auc)

        # Get the current loss for early stopping
        curr_loss = val_losses[loss_for_early_stopping][-1]

        # Remember the best epoch
        if best_epoch is None or best_loss is None or curr_loss < best_loss:
            best_epoch = epoch
            best_loss = curr_loss

        if time_for_early_stopping(val_losses[loss_for_early_stopping], look_behind):
            print("Training stopped due to low progress in validation loss!\n")
            break

        print()
        gc.collect()

    # Save losses to files
    train_losses_filename = os.path.join(dgcnn.model_output_dir, dgcnn.model, "Train_losses.csv")
    pd.DataFrame(train_losses).to_csv(train_losses_filename, index=False)

    val_losses_filename = os.path.join(dgcnn.model_output_dir, dgcnn.model, "Validation_losses.csv")
    pd.DataFrame(val_losses).to_csv(val_losses_filename, index=False)

    # Retrain the model on train + validation data
    train_validation_idxes = list(range(dgcnn.splits["Train"] + dgcnn.splits["Validation"]))
    for epoch in range(best_epoch+1):
        train_one_epoch(dgcnn, epoch, batch_size, train_validation_idxes, train_losses, "Train+Validation", print_auc)
        gc.collect()

    # Extract embedded features
    if extract_features:
        train_val_graphs = load_next_batch(dgcnn.cnf_dir,
                                           dgcnn.instance_ids,
                                           list(sorted(train_validation_idxes)),
                                           dgcnn.splits,
                                           "Train+Validation")
        features, labels = dgcnn.predictor.output_features(train_val_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt(os.path.join(dgcnn.model_output_dir, dgcnn.model, 'extracted_features_train.txt'),
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

    # Persist the model
    dgcnn.persist()


def test(dgcnn: DGCNNPredictor, batch_size: int, extract_features=False, print_auc=False):
    # Load previously persisted model
    dgcnn.load()

    # Test the final model
    dgcnn.predictor.eval()
    test_loss = loop_dataset(cnf_dir=dgcnn.cnf_dir,
                             model_output_dir=dgcnn.model_output_dir,
                             model=dgcnn.model,
                             instance_ids=dgcnn.instance_ids,
                             splits=dgcnn.splits,
                             epoch=0,
                             classifier=dgcnn.predictor,
                             sample_idxes=list(range(dgcnn.splits["Test"])),
                             optimizer=None,
                             batch_size=batch_size,
                             dataset_type="Test")
    if not print_auc:
        test_loss[2] = 0.0
    if dgcnn.regression:
        print('\033[92m  Average test of epoch %d: loss %.5f mae %.5f\033[0m' % (
            0, test_loss[0], test_loss[1]))
    else:
        print('\033[92m  Average test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
              0, test_loss[0], test_loss[1], test_loss[2]))

    if extract_features:
        test_graphs = load_next_batch(dgcnn.cnf_dir,
                                      dgcnn.instance_ids,
                                      list(range(dgcnn.splits["Test"])),
                                      dgcnn.splits,
                                      "Test")
        features, labels = dgcnn.predictor.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt(os.path.join(dgcnn.model_output_dir, dgcnn.model, 'extracted_features_test.txt'),
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
