import os

import pandas as pd
from sklearn import preprocessing

from .process_instances import prepare_features_data, prepare_output_data, filter_data_by_split


def load_data(data_dir: str):
    print("Preparing the X and Y data")

    splits = pd.read_csv(os.path.join(data_dir, "splits.csv"))

    prepare_features_data(data_dir, splits)
    x = pd.read_csv(os.path.join(data_dir, "all_data_x.csv"))

    prepare_output_data(data_dir, splits)
    y = pd.read_csv(os.path.join(data_dir, "all_data_y.csv"))

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
