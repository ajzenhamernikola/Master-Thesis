from __future__ import absolute_import

import math
import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm

from .DGCNN_SRC.DGCNN_embedding import DGCNN
from .DGCNN_SRC.mlp_dropout import MLPClassifier, MLPRegression
from .DGCNN_SRC.util import cmd_args, load_data


class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()
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

    def PrepareFeatureLabel(self, batch_graph):
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
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels


def loop_dataset(epoch, g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, dataset_type="train"):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].labels for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            predicted = pred.cpu().detach()
            print(predicted)
            all_scores.append(predicted)  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    np.savetxt(f'dgcnn_predictions/{dataset_type}_{epoch}_scores.txt', all_scores)  # output predictions

    if not classifier.regression and cmd_args.printAUC:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    else:
        avg_loss = np.concatenate((avg_loss, [0.0]))

    return avg_loss


def train_test():
    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier(regression=True)
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    train_losses = {"mse": [], "mae": []}
    test_losses = {"mse": [], "mae": []}
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(epoch, train_graphs, classifier, train_idxes, optimizer=optimizer, dataset_type="train")
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        if classifier.regression:
            print('\033[92maverage training of epoch %d: loss %.5f mae %.5f\033[0m' % (
                epoch, avg_loss[0], avg_loss[1]))
            train_losses["mse"].append(avg_loss[0])
            train_losses["mae"].append(avg_loss[1])
        else:
            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
                  epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        classifier.eval()
        test_loss = loop_dataset(epoch, test_graphs, classifier, list(range(len(test_graphs))), dataset_type="test")
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        if classifier.regression:
            print('\033[92maverage test of epoch %d: loss %.5f mae %.5f\033[0m' % (
                epoch, test_loss[0], test_loss[1]))
            test_losses["mse"].append(test_loss[0])
            test_losses["mae"].append(test_loss[1])
        else:
            print('\033[92maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
                  epoch, test_loss[0], test_loss[1], test_loss[2]))

    with open(cmd_args.data + '_acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    pd.DataFrame(train_losses).to_csv("dgcnn_train_losses.csv", index=False)
    pd.DataFrame(test_losses).to_csv("dgcnn_test_losses.csv", index=False)

    if cmd_args.printAUC:
        with open(cmd_args.data + '_auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    if cmd_args.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt',
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt',
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

    torch.save(classifier, "models/dgcnn")


if __name__ == "__main__":
    train_test()

    # Plot losses
    train_losses = pd.read_csv("dgcnn_train_losses.csv")
    test_losses = pd.read_csv("dgcnn_test_losses.csv")

    fig, (top_ax, bot_ax) = plt.subplots(2)
    fig.suptitle("Training/Test progress")
    fig.set_size_inches(w=len(train_losses) * 0.5, h=15)

    top_ax.set_ylabel("Loss value")
    top_ax.plot(range(len(train_losses["mse"])), train_losses["mse"], color="blue", linestyle="solid", label="mse")
    top_ax.plot(range(len(train_losses["mae"])), train_losses["mae"], color="red", linestyle="solid", label="mae")

    bot_ax.set_ylabel("Loss value")
    bot_ax.plot(range(len(test_losses["mse"])), test_losses["mse"], color="magenta", linestyle="solid", label="mse")
    bot_ax.plot(range(len(test_losses["mae"])), test_losses["mae"], color="orange", linestyle="solid", label="mae")

    for ax in fig.get_axes():
        ax.set_xticks(range(len(train_losses)))
        ax.set_xticklabels(range(1, len(train_losses) + 1))
        ax.set_xlabel("Epoch #")
        ax.legend()

    plt.savefig('dgcnn_predictions/graph_losses.png')
    plt.close()

    # Best model predictions
    best_epoch = np.argmin(test_losses["mse"])
    y_pred = np.loadtxt(f"dgcnn_predictions/test_{best_epoch}_scores.txt")
    y_true = np.loadtxt(f"src/models/DGCNN_SRC/data/{cmd_args.data}/test_ytrue.txt")
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

    png_file = 'dgcnn_predictions/graph_scores_per_solver.png'
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
