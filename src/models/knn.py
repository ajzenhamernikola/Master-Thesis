import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import metrics
from sklearn import multioutput

model_abbreviation = "KNN"
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "INSTANCES")
model_dir = os.path.join(data_dir, "..", "models", model_abbreviation)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def log10_transform_data(data):
    minimum_log10_value = 0.001
    data[data < minimum_log10_value] = minimum_log10_value
    return np.log10(data)


def filter_by_available_instances(available_instances: list):
    def closure(df: pd.DataFrame):
        filter_array = []
        for i in range(len(df)):
            value = df.iloc[i]["instance_id"] in available_instances
            filter_array.append(value)
        return filter_array
    return closure


def prepare_features_data(splits: pd.DataFrame):
    global data_dir
    all_data_x_file = os.path.join(data_dir, "chosen_data", "all_data_x.csv")
    if os.path.exists(all_data_x_file):
        return

    instance_ids = []
    for basedir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".features"):
                instance_id = os.path.relpath(os.path.join(basedir, file), data_dir)
                instance_ids.append(instance_id)

    xs = []
    for i in range(len(instance_ids)):
        file = instance_ids[i]
        n_series = pd.Series([file[:-9]], name="instance_id")
        data = pd.read_csv(os.path.join(data_dir, file))
        x = pd.merge(n_series, data, left_index=True, right_index=True)
        xs.append(x)

    x = pd.concat(xs)

    available_instances = list(splits["instance_id"])
    x = x.loc[filter_by_available_instances(available_instances), :]
    x = x.sort_values(by=["instance_id"])
    x.to_csv(all_data_x_file, index=False)


def prepare_output_data(splits: pd.DataFrame):
    global data_dir
    all_data_y_file = os.path.join(data_dir, "chosen_data", "all_data_y.csv")
    if os.path.exists(all_data_y_file):
        return

    data = pd.concat(
        [pd.read_csv(os.path.join(data_dir, "SAT12-HAND.csv")), pd.read_csv(os.path.join(data_dir, "SAT12-INDU.csv"))])

    instance_ids = []
    instance_id = None
    n = len(data)
    new_row = None
    columns = None
    ys = []
    for i in range(n):
        old_row = data.iloc[i]
        inst_id = old_row["instance_id"]
        if inst_id != instance_id:
            if new_row is not None:
                if inst_id in instance_ids:
                    continue
                y = pd.DataFrame([new_row], columns=columns)
                ys.append(y)
                instance_ids.append(instance_id)

            new_row = [inst_id]
            columns = ["instance_id"]
            instance_id = inst_id
        new_row.append(old_row["runtime"])
        columns.append(old_row["solver name"])

    y = pd.concat(ys)

    available_instances = list(splits["instance_id"])
    y = y.loc[filter_by_available_instances(available_instances), :]
    y = y.sort_values(by=["instance_id"])

    # Save all y data
    y.to_csv(all_data_y_file, index=False)


def filter_data_by_split(x_features: pd.DataFrame, y_outputs: pd.DataFrame, splits: pd.DataFrame, split: str):
    split_criteria = split.split("+")
    chosen_instances = []
    for i in range(len(splits)):
        instance_split = splits.iloc[i]["split"]
        if instance_split in split_criteria:
            instance_id = splits.iloc[i]["instance_id"]
            chosen_instances.append(instance_id)

    filter_indices = []
    for i in range(len(x_features)):
        instance_id = x_features.iloc[i]["instance_id"]
        filter_indices.append(instance_id in chosen_instances)

    x_result = x_features[filter_indices]
    x_result.index = pd.Series(list(range(len(chosen_instances))))

    filter_indices = []
    for i in range(len(y_outputs)):
        instance_id = y_outputs.iloc[i]["instance_id"]
        filter_indices.append(instance_id in chosen_instances)

    y_result = y_outputs[filter_indices]
    y_result.index = pd.Series(list(range(len(chosen_instances))))

    return x_result.drop(columns=["instance_id"]), log10_transform_data(y_result.drop(columns=["instance_id"]))


def load_data():
    print("Preparing the X and Y data")

    splits = pd.read_csv(os.path.join(data_dir, "chosen_data", "splits.csv"))

    prepare_features_data(splits)
    x = pd.read_csv(os.path.join(data_dir, "chosen_data", "all_data_x.csv"))

    prepare_output_data(splits)
    y = pd.read_csv(os.path.join(data_dir, "chosen_data", "all_data_y.csv"))

    x_train, y_train = filter_data_by_split(x, y, splits, "Train")
    x_val, y_val = filter_data_by_split(x, y, splits, "Validation")
    x_train_val, y_train_val = filter_data_by_split(x, y, splits, "Train+Validation")
    x_test, y_test = filter_data_by_split(x, y, splits, "Test")

    return x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test, y_test


def scale_the_data(x_train, x_val, x_train_val, x_test):
    print("Scaling the X and Y data")

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train_val)
    x_train_val = scaler.transform(x_train_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_train_val, x_test


def search_for_the_best_model(x_train, y_train, x_val, y_val, n_neighbors, weights, algorithm):
    print("Searching for the best model")
    number_of_solvers = y_train.shape[1]

    # Validation scores
    shape = (len(n_neighbors), len(weights), len(algorithm), number_of_solvers)
    r2_scores = np.empty(shape)
    rmse_scores = np.empty(shape)

    model_search_results = os.path.join(model_dir, f"{model_abbreviation}_model_search_results.csv")
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


def save_training_data(r2_scores, rmse_scores, n_neighbors, weights, algorithm, solver_names):
    print("Saving the training results")
    number_of_solvers = len(solver_names)

    model_search_results = os.path.join(model_dir, f"{model_abbreviation}_model_search_results.csv")
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

    model_search_group_results = os.path.join(model_dir, f"{model_abbreviation}_model_search_group_results.csv")
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


def retrain_the_best_model(x_train_val, y_train_val):
    print("Retraining the best model")

    model_filepath = os.path.join(model_dir, f"best_{model_abbreviation}_model")
    if os.path.exists(model_filepath):
        with open(model_filepath, "rb") as model_file:
            best_model = pickle.load(model_file)
            return best_model

    train_data = pd.read_csv(os.path.join(model_dir, f"{model_abbreviation}_model_search_group_results.csv"))
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


def evaluate_the_best_model(best_model, x_test, y_test):
    print("Evaluating the best model")
    number_of_solvers = y_test.shape[1]

    y_true, y_pred = y_test, best_model.predict(x_test)
    r2_scores_test = np.empty((number_of_solvers,))
    rmse_scores_test = np.empty((number_of_solvers,))
    for i in range(number_of_solvers):
        r2_scores_test[i] = metrics.r2_score(y_true.iloc[:, i:i + 1], y_pred[:, i:i + 1])
        rmse_scores_test[i] = metrics.mean_squared_error(y_true.iloc[:, i:i + 1], y_pred[:, i:i + 1], squared=False)

    return r2_scores_test, rmse_scores_test


def plot_the_data(r2_scores_test, rmse_scores_test, solver_names):
    print("Plotting the data")

    r2_score_test_avg = np.average(r2_scores_test)
    rmse_score_test_avg = np.average(rmse_scores_test)

    print(f"Average R2 score: {r2_score_test_avg}, Average RMSE score: {rmse_score_test_avg}")

    png_file = os.path.join(model_dir, f"{model_abbreviation}.png")
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


def save_the_best_model(best_model):
    print(f"Saving the best {model_abbreviation} model to a file")

    model_filepath = os.path.join(model_dir, f"best_{model_abbreviation}_model")
    if os.path.exists(model_filepath):
        return

    with open(model_filepath, "wb") as model_file:
        pickle.dump(best_model, model_file)


def main():
    x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test, y_test = load_data()
    x_train, x_val, x_train_val, x_test = scale_the_data(x_train, x_val, x_train_val, x_test)
    solver_names = y_train.columns

    n_neighbors = np.arange(1, 11)
    weights = ["uniform", "distance"]
    algorithm = ["ball_tree", "kd_tree", "brute"]

    r2_scores, rmse_scores = search_for_the_best_model(x_train, y_train, x_val, y_val, n_neighbors, weights, algorithm)
    save_training_data(r2_scores, rmse_scores, n_neighbors, weights, algorithm, solver_names)

    best_model = retrain_the_best_model(x_train_val, y_train_val)
    r2_scores_test, rmse_scores_test = evaluate_the_best_model(best_model, x_test, y_test)
    plot_the_data(r2_scores_test, rmse_scores_test, solver_names)
    save_the_best_model(best_model)


if __name__ == "__main__":
    main()
    print("Done")
