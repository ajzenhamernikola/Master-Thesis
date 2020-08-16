import os
import pickle as pkl

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics


def save_the_best_model(best_model, model_output_dir, model):
    print(f"Saving the best KNN model to a file")

    model_filepath = os.path.join(model_output_dir, f"best_{model}_model")
    if os.path.exists(model_filepath):
        return

    with open(model_filepath, "wb") as model_file:
        pkl.dump(best_model, model_file)


def calculate_r2_and_rmse_metrics(best_model, x_test, y_test, y_pred=None):
    if x_test is None and y_pred is None:
        raise ValueError("You must pass either x_test or y_pred")
    if x_test is not None and y_pred is not None:
        raise ValueError("You cannot pass both x_test and y_pred")

    print("Evaluating the best model")
    number_of_solvers = y_test.shape[1]
    r2_scores_test = np.empty((number_of_solvers,))
    rmse_scores_test = np.empty((number_of_solvers,))

    y_true = y_test
    if str(type(y_true)).find("DataFrame") != -1:
        y_true = y_true.values

    if y_pred is None:
        y_pred = best_model.predict(x_test)

    for i in range(number_of_solvers):
        r2_scores_test[i] = metrics.r2_score(y_true[:, i:i + 1], y_pred[:, i:i + 1])
        rmse_scores_test[i] = metrics.mean_squared_error(y_true[:, i:i + 1], y_pred[:, i:i + 1], squared=False)

    return np.average(r2_scores_test), np.average(rmse_scores_test), r2_scores_test, rmse_scores_test


def calculate_r2_and_rmse_metrics_nn(best_model, model_output_dir, model):
    y_pred = np.loadtxt(os.path.join(model_output_dir, "Test_ypred.txt"))
    y_true = np.loadtxt(os.path.join(model_output_dir, "Test_ytrue.txt"))

    return calculate_r2_and_rmse_metrics(best_model, None, y_true, y_pred)


def plot_r2_and_rmse_scores(r2_scores_test, rmse_scores_test, solver_names, model_output_dir, model):
    print("Plotting the data")

    r2_score_test_avg = np.average(r2_scores_test)
    rmse_score_test_avg = np.average(rmse_scores_test)

    print(f"Average R2 score: {r2_score_test_avg}, Average RMSE score: {rmse_score_test_avg}")

    png_file = os.path.join(model_output_dir, f"{model}.png")
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    xticks = range(1, len(r2_scores_test) + 1)
    yticks = np.round(r2_scores_test, 1)
    ymin = np.maximum(np.minimum(np.min(yticks), 0), -10)
    yticks = list(np.linspace(ymin, 1, int(10 * (1 - ymin)) + 1))
    ylabels = list(np.round(yticks, 1))
    ylim = (np.min(yticks), np.max(yticks))
    plt.title("R2 scores per solver")
    plt.xticks(ticks=xticks, labels=list(solver_names), rotation=90)
    plt.yticks(ticks=ylabels, labels=ylabels)
    plt.ylim(ylim)
    plt.bar(xticks, np.clip(r2_scores_test, ylim[0], ylim[1]), color="#578FF7")
    plt.plot([xticks[0], xticks[-1]], [r2_score_test_avg, r2_score_test_avg], "r-")
    
    plt.subplot(1, 2, 2)
    xticks = range(1, len(rmse_scores_test) + 1)
    rmse_score_test_max = np.ceil(np.max(rmse_scores_test))
    yticks = np.linspace(0, rmse_score_test_max, int(10*rmse_score_test_max) + 1)
    ylabels = np.round(np.linspace(0, rmse_score_test_max, int(10*rmse_score_test_max) + 1), 1)
    ylim = (np.min(yticks), np.max(yticks))
    plt.title("RMSE scores per solver")
    plt.xticks(ticks=xticks, labels=list(solver_names), rotation=90)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.ylim(ylim)
    plt.bar(xticks, np.clip(rmse_scores_test, ylim[0], ylim[1]), color="#FA6A68")
    plt.plot([xticks[0], xticks[-1]], [rmse_score_test_avg, rmse_score_test_avg], "b-")

    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    plt.close()


def plot_r2_and_rmse_scores_nn(r2_scores_test, rmse_scores_test, model_output_dir, model):
    solver_names = ["ebglucose", "ebminisat", "glucose2", "glueminisat", "lingeling", "lrglshr", "minisatpsm",
                    "mphaseSAT64", "precosat", "qutersat", "rcl", "restartsat", "cryptominisat2011", "spear-sw",
                    "spear-hw", "eagleup", "sparrow", "marchrw", "mphaseSATm", "satime11", "tnm", "mxc09", "gnoveltyp2",
                    "sattime", "sattimep", "clasp2", "clasp1", "picosat", "mphaseSAT", "sapperlot", "sol"]

    plot_r2_and_rmse_scores(r2_scores_test, rmse_scores_test, solver_names, model_output_dir, model)


def plot_losses_nn(model_output_dir, model):
    train_losses = pd.read_csv(os.path.join(model_output_dir, model, "Train_losses.csv"))
    val_losses = pd.read_csv(os.path.join(model_output_dir, model, "Validation_losses.csv"))
    
    n = len(train_losses)
    train_losses_mse = list(train_losses["mse"])
    train_losses_mae = list(train_losses["mae"])
    val_losses_mse = list(val_losses["mse"])
    val_losses_mae = list(val_losses["mae"])

    fig, (top_ax, bot_ax) = plt.subplots(2)
    fig.suptitle("Training/Validation progress")
    fig.set_size_inches(w=n * 0.5, h=15)

    top_ax.set_ylabel("MSE loss value")
    top_ax.plot(range(n), train_losses_mse, color="blue", linestyle="solid", label="Train")
    top_ax.plot(range(n), val_losses_mse, color="magenta", linestyle="solid", label="Val")

    bot_ax.set_ylabel("MAE loss value")
    bot_ax.plot(range(n), train_losses_mae, color="red", linestyle="solid", label="Train")
    bot_ax.plot(range(n), val_losses_mae, color="orange", linestyle="solid", label="Val")

    for ax in fig.get_axes():
        ax.set_xticks(range(n))
        ax.set_xticklabels(range(1, n + 1))
        ax.set_xlabel("Epoch #")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, model, f"{model}_losses.png"))
    plt.close()
