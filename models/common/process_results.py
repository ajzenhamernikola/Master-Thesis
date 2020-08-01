import os
import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def save_the_best_model(best_model, model_dir, model):
    print(f"Saving the best KNN model to a file")

    model_filepath = os.path.join(model_dir, f"best_{model}_model")
    if os.path.exists(model_filepath):
        return

    with open(model_filepath, "wb") as model_file:
        pkl.dump(best_model, model_file)


def calculate_r2_and_rmse_metrics(best_model, x_test, y_test):
    print("Evaluating the best model")
    number_of_solvers = y_test.shape[1]

    y_true, y_pred = y_test, best_model.predict(x_test)
    r2_scores_test = np.empty((number_of_solvers,))
    rmse_scores_test = np.empty((number_of_solvers,))
    for i in range(number_of_solvers):
        r2_scores_test[i] = metrics.r2_score(y_true.iloc[:, i:i + 1], y_pred[:, i:i + 1])
        rmse_scores_test[i] = metrics.mean_squared_error(y_true.iloc[:, i:i + 1], y_pred[:, i:i + 1], squared=False)

    return r2_scores_test, rmse_scores_test


def plot_the_data(r2_scores_test, rmse_scores_test, solver_names, model_dir, model):
    print("Plotting the data")

    r2_score_test_avg = np.average(r2_scores_test)
    rmse_score_test_avg = np.average(rmse_scores_test)

    print(f"Average R2 score: {r2_score_test_avg}, Average RMSE score: {rmse_score_test_avg}")

    png_file = os.path.join(model_dir, f"{model}.png")
    if os.path.exists(png_file):
        return

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    xticks = range(1, len(r2_scores_test) + 1)
    yticks = np.linspace(0, 1, 10)
    ylabels = np.round(np.linspace(0, 1, 10), 1)
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
