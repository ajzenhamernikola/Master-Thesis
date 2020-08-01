import os
import pickle
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import metrics
from sklearn import multioutput


def search_for_the_best_model(x_train, y_train, x_val, y_val, n_neighbors, weights, algorithm, model_dir):
    print("Searching for the best model")
    number_of_solvers = y_train.shape[1]

    # Validation scores
    shape = (len(n_neighbors), len(weights), len(algorithm), number_of_solvers)
    r2_scores = np.empty(shape)
    rmse_scores = np.empty(shape)

    model_search_results = os.path.join(model_dir, f"KNN_model_search_results.csv")
    if os.path.exists(model_search_results):
        data = pd.read_csv(model_search_results)
        i_mult = len(weights) * len(algorithm) * number_of_solvers
        j_mult = len(algorithm) * number_of_solvers
        k_mult = number_of_solvers

        for i in range(len(n_neighbors)):
            for j in range(len(weights)):
                for k in range(len(algorithm)):
                    for l in range(number_of_solvers):
                        idx = i * i_mult + k * k_mult + j * j_mult + l
                        r2_score = data.iloc[idx]["r2 score"]
                        r2_scores[i, j, k, l] = r2_score
                        rmse_score = data.iloc[idx]["rmse score"]
                        rmse_scores[i, j, k, l] = rmse_score

        return r2_scores, rmse_scores

    max_count = np.product(shape)
    count = 0

    # Searching procedure
    for i in range(len(n_neighbors)):
        param_n_neighbors = n_neighbors[i]
        for j in range(len(weights)):
            param_weights = weights[j]
            for k in range(len(algorithm)):
                param_algorithm = algorithm[k]
                for l in range(number_of_solvers):
                    count += 1
                    print(f"\tTraining model {count}/{max_count}")
                    model = neighbors.KNeighborsRegressor(
                        n_neighbors=param_n_neighbors, weights=param_weights, algorithm=param_algorithm, n_jobs=-1)
                    model.fit(x_train, y_train.iloc[:, l:l + 1])
                    y_true, y_pred = y_val.iloc[:, l:l + 1], model.predict(x_val)
                    r2_score = metrics.r2_score(y_true, y_pred)
                    r2_scores[i, j, k, l] = r2_score
                    rmse_score = metrics.mean_squared_error(y_true, y_pred, squared=False)
                    rmse_scores[i, j, k, l] = rmse_score

    return r2_scores, rmse_scores


def save_training_data(r2_scores, rmse_scores, n_neighbors, weights, algorithm, solver_names, model_dir):
    print("Saving the training results")
    number_of_solvers = len(solver_names)

    model_search_results = os.path.join(model_dir, f"KNN_model_search_results.csv")
    if not os.path.exists(model_search_results):
        with open(model_search_results, "w", encoding="utf-8") as csv:
            csv.write("n_neighbors,weights,algorithm,solver name,r2 score,rmse score\n")
            for i in range(len(n_neighbors)):
                param_n_neighbors = n_neighbors[i]
                for j in range(len(weights)):
                    param_weights = weights[j]
                    for k in range(len(algorithm)):
                        param_algorithm = algorithm[k]
                        for l in range(number_of_solvers):
                            solver_name = solver_names[l]
                            row = f"{param_n_neighbors},{param_weights},{param_algorithm},{solver_name},{r2_scores[i, j, k, l]},{rmse_scores[i, j, k, l]}\n"
                            csv.write(row)

    model_search_group_results = os.path.join(model_dir, f"KNN_model_search_group_results.csv")
    if not os.path.exists(model_search_group_results):
        with open(model_search_group_results, "w", encoding="utf-8") as csv:
            csv.write(
                "n_neighbors,weights,algorithm,avg r2 score,min r2 score,max r2 score,avg rmse score,min rmse score,max rmse score\n")
            for i in range(len(n_neighbors)):
                param_n_neighbors = n_neighbors[i]
                for j in range(len(weights)):
                    param_weights = weights[j]
                    for k in range(len(algorithm)):
                        param_algorithm = algorithm[k]
                        avg_r2_score = np.average(r2_scores[i, j, k])
                        min_r2_score = np.min(r2_scores[i, j, k])
                        max_r2_score = np.max(r2_scores[i, j, k])
                        avg_rmse_score = np.average(rmse_scores[i, j, k])
                        min_rmse_score = np.min(rmse_scores[i, j, k])
                        max_rmse_score = np.max(rmse_scores[i, j, k])
                        row = f"{param_n_neighbors},{param_weights},{param_algorithm},{avg_r2_score},{min_r2_score},{max_r2_score},{avg_rmse_score},{min_rmse_score},{max_rmse_score}\n"
                        csv.write(row)


def retrain_the_best_model(x_train_val, y_train_val, model_dir):
    print("Retraining the best model")

    model_filepath = os.path.join(model_dir, f"best_KNN_model")
    if os.path.exists(model_filepath):
        with open(model_filepath, "rb") as model_file:
            best_model = pickle.load(model_file)
            return best_model

    train_data = pd.read_csv(os.path.join(model_dir, f"KNN_model_search_group_results.csv"))
    avg_r2_score = train_data["avg r2 score"]
    idx = np.argmax(avg_r2_score)

    best_data = train_data.iloc[idx]
    best_params = {
        "n_neighbors": int(best_data["n_neighbors"]),
        "weights": best_data["weights"],
        "algorithm": best_data["algorithm"]
    }
    print(f"\tBest params: {best_params}")
    best_model = multioutput.MultiOutputRegressor(neighbors.KNeighborsRegressor(**best_params, n_jobs=-1))
    best_model.fit(x_train_val, y_train_val)

    return best_model


def train(x_train, y_train, x_val, y_val, solver_names, model_dir):
    n_neighbors = np.arange(1, 11)
    weights = ["uniform", "distance"]
    algorithm = ["ball_tree", "kd_tree", "brute"]

    r2_scores, rmse_scores = search_for_the_best_model(x_train, y_train, x_val, y_val, n_neighbors, weights, algorithm, model_dir)
    save_training_data(r2_scores, rmse_scores, n_neighbors, weights, algorithm, solver_names, model_dir)
