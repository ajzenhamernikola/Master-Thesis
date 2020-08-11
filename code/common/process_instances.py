import os

import pandas as pd

from .math import log10_transform_data


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