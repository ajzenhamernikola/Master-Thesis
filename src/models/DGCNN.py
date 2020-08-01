import gc
import random
import sys
import os
import pickle as pkl

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

from .DGCNN_SRC.DGCNN_embedding import DGCNN
from .DGCNN_SRC.mlp_dropout import MLPClassifier, MLPRegression
from .DGCNN_SRC.util import cmd_args, pickle_data, loop_dataset, load_next_batch, time_for_early_stopping


class Predictor(nn.Module):
    def __init__(self, regression=False):
        super(Predictor, self).__init__()
        self.regression = regression
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim + cmd_args.attr_dim,
                             num_edge_feats=cmd_args.edge_feat_dim,
                             k=cmd_args.sortpooling_k,
                             conv1d_activation=cmd_args.conv1d_activation)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim

        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, output_size=cmd_args.num_class,
                                     with_dropout=cmd_args.dropout)
        else:
            self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class,
                                     with_dropout=cmd_args.dropout)

    def prepare_feature_labels(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph), len(batch_graph[0].labels))
        else:
            labels = torch.LongTensor(len(batch_graph), len(batch_graph[0].labels))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = torch.FloatTensor(batch_graph[i].labels)
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif not node_feat_flag and node_tag_flag:
            node_feat = node_tag
        elif node_feat_flag and not node_tag_flag:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if edge_feat_flag:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph):
        embed, labels = self.output_features(batch_graph)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.prepare_feature_labels(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels


def plot_losses():
    train_losses = pd.read_csv("models/DGCNN/dgcnn_Train_losses.csv")
    val_losses = pd.read_csv("models/DGCNN/dgcnn_Validation_losses.csv")

    fig, (top_ax, bot_ax) = plt.subplots(2)
    fig.suptitle("Training/Validation progress")
    fig.set_size_inches(w=len(train_losses) * 0.5, h=15)

    top_ax.set_ylabel("MSE loss value")
    top_ax.plot(range(len(train_losses["mse"])), train_losses["mse"], color="blue", linestyle="solid", label="Train")
    top_ax.plot(range(len(val_losses["mse"])), val_losses["mse"], color="magenta", linestyle="solid", label="Val")

    bot_ax.set_ylabel("MAE loss value")
    bot_ax.plot(range(len(train_losses["mae"])), train_losses["mae"], color="red", linestyle="solid", label="Train")
    bot_ax.plot(range(len(val_losses["mae"])), val_losses["mae"], color="orange", linestyle="solid", label="Val")

    for ax in fig.get_axes():
        ax.set_xticks(range(len(train_losses)))
        ax.set_xticklabels(range(1, len(train_losses) + 1))
        ax.set_xlabel("Epoch #")
        ax.legend()

    plt.savefig('models/DGCNN/graph_losses.png')
    plt.close()


def plot_scores_per_solver():
    # Best model predictions
    y_pred = np.loadtxt(f"models/DGCNN/Test_0_scores.txt")
    y_true = np.loadtxt(f"INSTANCES/DGCNN/{cmd_args.data}/test_ytrue.txt")
    number_of_solvers = y_pred.shape[1]

    # Plot scores per solver
    r2_scores_test = np.empty((number_of_solvers,))
    rmse_scores_test = np.empty((number_of_solvers,))
    for i in range(number_of_solvers):
        r2_scores_test[i] = metrics.r2_score(y_true[:, i:i + 1], y_pred[:, i:i + 1])
        rmse_scores_test[i] = metrics.mean_squared_error(y_true[:, i:i + 1], y_pred[:, i:i + 1], squared=False)

    r2_score_test_avg = np.average(r2_scores_test)
    rmse_score_test_avg = np.average(rmse_scores_test)

    print(f"Average R2 score: {r2_score_test_avg}, Average RMSE score: {rmse_score_test_avg}")

    png_file = 'models/DGCNN/graph_scores_per_solver.png'
    solver_names = ["ebglucose", "ebminisat", "glucose2", "glueminisat", "lingeling", "lrglshr", "minisatpsm",
                    "mphaseSAT64", "precosat", "qutersat", "rcl", "restartsat", "cryptominisat2011", "spear-sw",
                    "spear-hw", "eagleup", "sparrow", "marchrw", "mphaseSATm", "satime11", "tnm", "mxc09", "gnoveltyp2",
                    "sattime", "sattimep", "clasp2", "clasp1", "picosat", "mphaseSAT", "sapperlot", "sol"]

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    xticks = range(1, len(r2_scores_test) + 1)
    ymin = int(np.floor(np.min(r2_scores_test)))
    yticks = np.linspace(ymin, 1, 10 * (1 - ymin))
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


class DGCNNPredictor(object):
    def __init__(self):
        # Inits
        self.predictor = None
        self.last_trained_predictor = None
        self.model_filename = "models/DGCNN/best_DGCNN_model"

        self.data_dir = os.path.join(".", "INSTANCES")
        instance_ids_filename = os.path.join(self.data_dir, "DGCNN", cmd_args.data, "instance_ids.pickled")
        with open(instance_ids_filename, "rb") as f:
            instances_metadata = pkl.load(f)
            self.instance_ids = instances_metadata[0]
            self.splits = instances_metadata[1]

        pickle_data(self.data_dir, self.instance_ids, self.splits)

    def train(self):
        overwrite = True
        if os.path.exists(self.model_filename):
            print("\nModel already exist. Would you like to overwrite it [o], continue training [t], " +
                  "or use existing [e]?")
            response = input().lower()
            if response == "e":
                return
            if response == "t":
                overwrite = False
            elif response != "o":
                raise ValueError(f"You responded with '{response}', while the options are one of {{'o', 't', 'e'}}")

        if overwrite:
            self.predictor = Predictor(regression=True)
        else:
            self.predictor = torch.load(self.model_filename)

        if cmd_args.mode == 'gpu':
            self.predictor = self.predictor.cuda()
            print("Optimizing on a GPU\n")
        else:
            print("Optimizing on a CPU\n")

        optimizer = optim.Adam(self.predictor.parameters(), lr=cmd_args.learning_rate)

        train_idxes = list(range(self.splits["Train"]))

        best_loss = None
        best_epoch = None
        train_losses = {"mse": [], "mae": []}
        val_losses = {"mse": [], "mae": []}

        loss_for_early_stopping = "mse"
        look_behind = 20
        for epoch in range(cmd_args.num_epochs):
            # Train one epoch
            random.shuffle(train_idxes)
            self.predictor.train()
            avg_loss = loop_dataset(data_dir=self.data_dir,
                                    instance_ids=self.instance_ids,
                                    splits=self.splits,
                                    epoch=epoch,
                                    classifier=self.predictor,
                                    sample_idxes=train_idxes,
                                    optimizer=optimizer,
                                    bsize=cmd_args.batch_size,
                                    dataset_type="Train")
            if not cmd_args.printAUC:
                avg_loss[2] = 0.0
            if self.predictor.regression:
                print('\033[92m  Average training of epoch %d: loss %.5f mae %.5f\033[0m' % (
                    epoch, avg_loss[0], avg_loss[1]))
                train_losses["mse"].append(avg_loss[0])
                train_losses["mae"].append(avg_loss[1])
            else:
                print('\033[92m  Average training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
                      epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

            # Validate one epoch
            self.predictor.eval()
            val_loss = loop_dataset(data_dir=self.data_dir,
                                    instance_ids=self.instance_ids,
                                    splits=self.splits,
                                    epoch=epoch,
                                    classifier=self.predictor,
                                    sample_idxes=list(range(self.splits["Validation"])),
                                    optimizer=None,
                                    bsize=cmd_args.batch_size,
                                    dataset_type="Validation")
            if not cmd_args.printAUC:
                val_loss[2] = 0.0
            if self.predictor.regression:
                print('\033[92m  Average validation of epoch %d: loss %.5f mae %.5f\033[0m' % (
                    epoch, val_loss[0], val_loss[1]))
                val_losses["mse"].append(val_loss[0])
                val_losses["mae"].append(val_loss[1])
            else:
                print('\033[92m  Average validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
                    epoch, val_loss[0], val_loss[1], val_loss[2]))

            # Remember the last fully trained predictor
            self.last_trained_predictor = self.predictor

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

        pd.DataFrame(train_losses).to_csv("models/DGCNN/dgcnn_Train_losses.csv", index=False)
        pd.DataFrame(val_losses).to_csv("models/DGCNN/dgcnn_Validation_losses.csv", index=False)

        # Retrain the model on train + validation data
        train_validation_idxes = list(range(self.splits["Train"] + self.splits["Validation"]))
        for epoch in range(cmd_args.num_epochs):
            # Train one epoch
            random.shuffle(train_validation_idxes)
            self.predictor.train()
            avg_loss = loop_dataset(data_dir=self.data_dir,
                                    instance_ids=self.instance_ids,
                                    splits=self.splits,
                                    epoch=epoch,
                                    classifier=self.predictor,
                                    sample_idxes=train_validation_idxes,
                                    optimizer=optimizer,
                                    bsize=cmd_args.batch_size,
                                    dataset_type="Train+Validation")
            if not cmd_args.printAUC:
                avg_loss[2] = 0.0
            if self.predictor.regression:
                print('\033[92m  Average training of epoch %d: loss %.5f mae %.5f\033[0m' % (
                    epoch, avg_loss[0], avg_loss[1]))
            else:
                print('\033[92m  Average training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
                      epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

            # Remember the last fully trained predictor
            self.last_trained_predictor = self.predictor

        if cmd_args.extract_features:
            train_val_graphs = load_next_batch(self.data_dir,
                                               self.instance_ids,
                                               list(sorted(train_validation_idxes)),
                                               self.splits,
                                               "Train+Validation")
            features, labels = self.predictor.output_features(train_val_graphs)
            labels = labels.type('torch.FloatTensor')
            np.savetxt('models/DGCNN/extracted_features_train.txt',
                       torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

        torch.save(self.predictor, self.model_filename)

    def test(self):
        self.predictor = torch.load(self.model_filename)
        test_losses = {"mse": [], "mae": []}

        # Test the final model
        self.predictor.eval()
        test_loss = loop_dataset(data_dir=self.data_dir,
                                 instance_ids=self.instance_ids,
                                 splits=self.splits,
                                 epoch=0,
                                 classifier=self.predictor,
                                 sample_idxes=list(range(self.splits["Test"])),
                                 optimizer=None,
                                 bsize=cmd_args.batch_size,
                                 dataset_type="Test")
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        if self.predictor.regression:
            print('\033[92m  Average test of epoch %d: loss %.5f mae %.5f\033[0m' % (
                0, test_loss[0], test_loss[1]))
            test_losses["mse"].append(test_loss[0])
            test_losses["mae"].append(test_loss[1])
        else:
            print('\033[92m  Average test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
                  0, test_loss[0], test_loss[1], test_loss[2]))

        with open("models/DGCNN/" + cmd_args.data + '_acc_results.txt', 'a+') as f:
            f.write(str(test_loss[1]) + '\n')

        pd.DataFrame(test_losses).to_csv("models/DGCNN/dgcnn_Test_losses.csv", index=False)

        if cmd_args.printAUC:
            with open(cmd_args.data + '_auc_results.txt', 'a+') as f:
                f.write(str(test_loss[2]) + '\n')

        if cmd_args.extract_features:
            test_graphs = load_next_batch(self.data_dir,
                                          self.instance_ids,
                                          list(range(self.splits["Test"])),
                                          self.splits,
                                          "Test")
            features, labels = self.predictor.output_features(test_graphs)
            labels = labels.type('torch.FloatTensor')
            np.savetxt('models/DGCNN/extracted_features_test.txt',
                       torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

    def __del__(self):
        # In case a problem arises, serialize the last known fully trained predictor
        torch.save(self.last_trained_predictor, self.model_filename)


def main():
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    dgcnn = DGCNNPredictor()
    dgcnn.train()
    dgcnn.test()

    plot_losses()
    plot_scores_per_solver()


if __name__ == "__main__":
    main()
