import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import multioutput


def search_for_the_best_model(x_train, y_train, x_val, y_val, n_estimators, min_samples_splits, model_dir):
    print("Searching for the best model")
    number_of_solvers = y_train.shape[1]

    # Validation scores
    shape = (len(n_estimators), len(min_samples_splits), number_of_solvers)
    r2_scores = np.empty(shape)
    rmse_scores = np.empty(shape)

    model_search_results = os.path.join(model_dir, f"RF_model_search_results.csv")
    if os.path.exists(model_search_results):
        data = pd.read_csv(model_search_results)
        i_mult = len(min_samples_splits) * number_of_solvers
        j_mult = number_of_solvers

        for i in range(len(n_estimators)):
            for j in range(len(min_samples_splits)):
                for k in range(number_of_solvers):
                    idx = i * i_mult + j * j_mult + k
                    r2_score = data.iloc[idx]["r2 score"]
                    r2_scores[i, j, k] = r2_score
                    rmse_score = data.iloc[idx]["rmse score"]
                    rmse_scores[i, j, k] = rmse_score

        return r2_scores, rmse_scores

    max_count = np.product(shape)
    count = 0

    # Searching procedure
    for i in range(len(n_estimators)):
        param_n_estimator = n_estimators[i]
        for j in range(len(min_samples_splits)):
            param_min_samples_split = min_samples_splits[j]
            for k in range(number_of_solvers):
                count += 1
                print(f"\tTraining model {count}/{max_count}")
                model = RandomForestRegressor(n_estimators=param_n_estimator,
                                              min_samples_split=param_min_samples_split,
                                              n_jobs=-1)
                model.fit(x_train, y_train.iloc[:, k:k + 1].values.ravel())
                y_true, y_pred = y_val.iloc[:, k:k + 1], model.predict(x_val)
                r2_score = metrics.r2_score(y_true, y_pred)
                r2_scores[i, j, k] = r2_score
                rmse_score = metrics.mean_squared_error(y_true, y_pred, squared=False)
                rmse_scores[i, j, k] = rmse_score

    return r2_scores, rmse_scores


def save_training_data(r2_scores, rmse_scores, n_estimators, min_samples_splits, solver_names, model_dir):
    print("Saving the training results")
    number_of_solvers = len(solver_names)

    model_search_results = os.path.join(model_dir, f"RF_model_search_results.csv")
    if not os.path.exists(model_search_results):
        with open(model_search_results, "w", encoding="utf-8") as csv:
            csv.write("n_estimators,min_samples_split,solver name,r2 score,rmse score\n")
            for i in range(len(n_estimators)):
                param_n_estimator = n_estimators[i]
                for j in range(len(min_samples_splits)):
                    param_min_samples_split = min_samples_splits[j]
                    for k in range(number_of_solvers):
                        solver_name = solver_names[k]
                        row = f"{param_n_estimator},{param_min_samples_split}," + \
                              f"{solver_name},{r2_scores[i, j, k]},{rmse_scores[i, j, k]}\n"
                        csv.write(row)

    model_search_group_results = os.path.join(model_dir, f"RF_model_search_group_results.csv")
    if not os.path.exists(model_search_group_results):
        with open(model_search_group_results, "w", encoding="utf-8") as csv:
            csv.write("n_estimators,min_samples_split,avg r2 score,min r2 score,max r2 score," +
                      "avg rmse score,min rmse score,max rmse score\n")
            for i in range(len(n_estimators)):
                param_n_estimator = n_estimators[i]
                for j in range(len(min_samples_splits)):
                    param_min_samples_split = min_samples_splits[j]

                    avg_r2_score = np.average(r2_scores[i, j])
                    min_r2_score = np.min(r2_scores[i, j])
                    max_r2_score = np.max(r2_scores[i, j])
                    avg_rmse_score = np.average(rmse_scores[i, j])
                    min_rmse_score = np.min(rmse_scores[i, j])
                    max_rmse_score = np.max(rmse_scores[i, j])
                    row = f"{param_n_estimator},{param_min_samples_split}," + \
                          f"{avg_r2_score},{min_r2_score},{max_r2_score}," + \
                          f"{avg_rmse_score},{min_rmse_score},{max_rmse_score}\n"
                    csv.write(row)


def retrain_the_best_model(x_train_val, y_train_val, model_dir):
    print("Retraining the best model")

    model_filepath = os.path.join(model_dir, "best_RF_model")
    if os.path.exists(model_filepath):
        with open(model_filepath, "rb") as model_file:
            best_model = pickle.load(model_file)
            return best_model

    train_data = pd.read_csv(os.path.join(model_dir, f"RF_model_search_group_results.csv"))
    avg_r2_score = train_data["avg r2 score"]
    idx = np.argmax(avg_r2_score)

    best_data = train_data.iloc[idx]
    best_params = {
        "n_estimators": int(best_data["n_estimators"]),
        "min_samples_split": int(best_data["min_samples_split"]) if best_data["min_samples_split"] >= 1 else best_data["min_samples_split"]
    }

    print(f"\tBest params: {best_params}")
    best_model = multioutput.MultiOutputRegressor(RandomForestRegressor(**best_params, n_jobs=-1))
    best_model.fit(x_train_val, y_train_val)

    return best_model


def train(x_train, y_train, x_val, y_val, solver_names, model_dir):
    n_estimators = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    min_samples_splits = [0.05, 0.1, 0.2, 0.3, 2, 10, 20, 50]

    r2_scores, rmse_scores = search_for_the_best_model(x_train, y_train, x_val, y_val, n_estimators, min_samples_splits,
                                                       model_dir)
    save_training_data(r2_scores, rmse_scores, n_estimators, min_samples_splits, solver_names, model_dir)
