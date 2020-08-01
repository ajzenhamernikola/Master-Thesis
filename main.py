import os

from preprocessing.os.arguments import cmd_args
from models import knn, rf
from models.common.data import load_data, scale_the_data
from models.common.process_results import save_the_best_model, calculate_r2_and_rmse_metrics, plot_the_data


# Globals for KNN and RF models
x_train = None
y_train = None
x_val = None
y_val = None
x_train_val = None
y_train_val = None
x_test = None
y_test = None
solver_names = None
best_model = None
r2_scores_test = None
rmse_scores_test = None


def data_preparation():
    global x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test, y_test, solver_names

    if cmd_args.model == "KNN" or cmd_args.model == "RF":
        x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test, y_test = load_data(cmd_args.cnf_dir)
        x_train, x_val, x_train_val, x_test = scale_the_data(x_train, x_val, x_train_val, x_test)
        solver_names = y_train.columns
    else:
        pass


def train_model():
    global x_train, y_train, x_val, y_val, x_train_val, y_train_val, best_model

    if cmd_args.model == "KNN":
        knn.train(x_train, y_train, x_val, y_val, solver_names, cmd_args.model_dir)
        best_model = knn.retrain_the_best_model(x_train_val, y_train_val, cmd_args.model_dir)
    elif cmd_args.model == "RF":
        rf.train(x_train, y_train, x_val, y_val, solver_names, cmd_args.model_dir)
        best_model = rf.retrain_the_best_model(x_train_val, y_train_val, cmd_args.model_dir)
    else:
        pass

    if cmd_args.model == "KNN" or cmd_args.model == "RF":
        save_the_best_model(best_model, cmd_args.model_dir, cmd_args.model)


def evaluate_model():
    global x_test, y_test, best_model, r2_scores_test, rmse_scores_test

    if cmd_args.model == "KNN" or cmd_args.model == "RF":
        r2_scores_test, rmse_scores_test = calculate_r2_and_rmse_metrics(best_model, x_test, y_test)
    else:
        pass


def process_results():
    global r2_scores_test, rmse_scores_test, solver_names

    if cmd_args.model == "KNN" or cmd_args.model == "RF":
        plot_the_data(r2_scores_test, rmse_scores_test, solver_names, cmd_args.model_dir, cmd_args.model)
    else:
        pass


def main():
    # Creating the directory for model outputs
    model_dir = os.path.join(cmd_args.model_output_dir, cmd_args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    cmd_args.model_dir = model_dir

    # Train and evaluate chosen model
    data_preparation()
    train_model()
    evaluate_model()
    process_results()


if __name__ == "__main__":
    main()
