import random
import os
import gc

import numpy as np
import pandas as pd
import torch

from .libs.dgcnn.classes import DGCNNPredictor
from .libs.dgcnn.util import loop_dataset, load_next_batch
from .common.nn import time_for_early_stopping


def print_box(message: str, length=80):
    message_len = len(message)
    left_spaces = right_spaces = ((80 - message_len - 1) // 2)
    if message_len % 2 == 1:
        left_spaces -= 1
    print(length * "*")
    print("*" + left_spaces * " " + message + right_spaces * " " + "*")
    print(length * "*")


def train_one_epoch(dgcnn: DGCNNPredictor, epoch: int, batch_size: int, idxes: list, train_losses: dict,
                    dataset_type: str, print_auc: bool):
    dgcnn.predictor.train()
    avg_loss = loop_dataset(cnf_dir=dgcnn.cnf_dir,
                            model_output_dir=dgcnn.model_output_dir,
                            model=dgcnn.model,
                            instance_ids=dgcnn.instance_ids,
                            splits=dgcnn.splits,
                            epoch=epoch,
                            classifier=dgcnn.predictor,
                            sample_idxes=idxes,
                            random_shuffle=True,
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
    
    val_idxes = list(range(dgcnn.splits["Train"], dgcnn.splits["Train"] + dgcnn.splits["Validation"]))
    
    val_loss = loop_dataset(cnf_dir=dgcnn.cnf_dir,
                            model_output_dir=dgcnn.model_output_dir,
                            model=dgcnn.model,
                            instance_ids=dgcnn.instance_ids,
                            splits=dgcnn.splits,
                            epoch=epoch,
                            classifier=dgcnn.predictor,
                            sample_idxes=val_idxes,
                            random_shuffle=False,
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


def train(dgcnn: DGCNNPredictor, num_epochs: int, batch_size: int, look_behind: int, print_auc=False):
    train_idxes = list(range(dgcnn.splits["Train"]))
    loss_for_early_stopping = "mse"
    first_epoch = len(dgcnn.train_losses[loss_for_early_stopping])

    if first_epoch != 0:
        print(f"Enter the number of epochs to train after {first_epoch}:")
        additional_epochs = int(input().strip())
        if additional_epochs == 0:
            return
        look_behind += additional_epochs

    print_box("TRAINING")

    for epoch in range(first_epoch, num_epochs):
        # Train one epoch
        train_one_epoch(dgcnn, epoch, batch_size, train_idxes, dgcnn.train_losses, "Train", print_auc)

        # Validate one epoch
        validate_one_epoch(dgcnn, epoch, batch_size, dgcnn.val_losses, print_auc)

        # Get the current loss for early stopping
        curr_loss = dgcnn.val_losses[loss_for_early_stopping][-1]

        # Remember the best epoch
        if dgcnn.best_epoch is None or dgcnn.best_loss is None or curr_loss < dgcnn.best_loss:
            dgcnn.best_epoch = epoch
            dgcnn.best_loss = curr_loss

        if time_for_early_stopping(dgcnn.val_losses[loss_for_early_stopping], look_behind):
            print("\nTraining stopped due to low progress in validation loss!")
            print("Do you want to train more? [y/n]")
            stop_training = input().strip().lower() == "n"
            if stop_training:
                break

            print("Do you want to train more? [y/n]")
            print(f"Enter the number of epochs to train after {look_behind}:")
            additional_epochs = int(input().strip())
            look_behind += additional_epochs

        print()
        gc.collect()

    # Save losses to files
    train_losses_filename = os.path.join(dgcnn.model_output_dir, dgcnn.model, "Train_losses.csv")
    pd.DataFrame(dgcnn.train_losses).to_csv(train_losses_filename, index=False)

    val_losses_filename = os.path.join(dgcnn.model_output_dir, dgcnn.model, "Validation_losses.csv")
    pd.DataFrame(dgcnn.val_losses).to_csv(val_losses_filename, index=False)

    # Persist the model in case retraining fails
    dgcnn.persist()


def retrain(dgcnn: DGCNNPredictor, batch_size: int, extract_features=False, print_auc=False):
    print("Do you want to retrain the model? [y/n]")
    no_retraining = input().strip().lower() == "n"
    if no_retraining:
        return

    print_box(f"RETRAINING {dgcnn.best_epoch} EPOCH(S)")

    # Retrain the model on train + validation data
    train_validation_idxes = list(range(dgcnn.splits["Train"] + dgcnn.splits["Validation"]))
    for epoch in range(dgcnn.best_epoch + 1):
        train_one_epoch(dgcnn, epoch, batch_size, train_validation_idxes, dgcnn.train_losses, "Train+Validation",
                        print_auc)
        gc.collect()

    print()

    # Extract embedded features
    if extract_features:
        train_val_graphs = load_next_batch(cnf_dir=dgcnn.cnf_dir,
                                           instance_ids=dgcnn.instance_ids,
                                           selected_idx=list(sorted(train_validation_idxes)),
                                           splits=dgcnn.splits,
                                           dataset_type="Train+Validation")
        features, labels = dgcnn.predictor.output_features(train_val_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt(os.path.join(dgcnn.model_output_dir, dgcnn.model, 'extracted_features_train.txt'),
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

    # Persist the model
    dgcnn.persist()


def test(dgcnn: DGCNNPredictor, batch_size: int, extract_features=False, print_auc=False):
    # Load previously persisted model
    dgcnn.load()

    print_box("TESTING")

    test_idxes = list(range(dgcnn.splits["Train"] + dgcnn.splits["Validation"],
                            dgcnn.splits["Train"] + dgcnn.splits["Validation"] + dgcnn.splits["Test"]))

    # Test the final model
    dgcnn.predictor.eval()
    test_loss = loop_dataset(cnf_dir=dgcnn.cnf_dir,
                             model_output_dir=dgcnn.model_output_dir,
                             model=dgcnn.model,
                             instance_ids=dgcnn.instance_ids,
                             splits=dgcnn.splits,
                             epoch=0,
                             classifier=dgcnn.predictor,
                             sample_idxes=test_idxes,
                             random_shuffle=False,
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
        test_graphs = load_next_batch(cnf_dir=dgcnn.cnf_dir,
                                      instance_ids=dgcnn.instance_ids,
                                      selected_idx=list(range(dgcnn.splits["Test"])),
                                      splits=dgcnn.splits,
                                      dataset_type="Test")
        features, labels = dgcnn.predictor.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt(os.path.join(dgcnn.model_output_dir, dgcnn.model, 'extracted_features_test.txt'),
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

    print()
