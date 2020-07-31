import os
import pickle as pkl
import pandas as pd
import numpy as np

from ..os.path import \
    collect_files_and_sizes
from ..os.process import \
    start_process
from ..csv.series import \
    get_column_without_duplicates
from ..parsers.libparsers import \
    PARSERSLIB


def collect_instance_names(csv_filename):
    data = pd.read_csv(csv_filename)
    col = get_column_without_duplicates(data, 'instance_id')
    col.sort()
    return col


def collect_cnf_files_and_sizes(directory, inst_dict):
    return collect_files_and_sizes(directory, '.cnf', inst_dict)


def calculate_numbers_of_variables_and_clauses(cnf_files):
    variables_dist = []
    clauses_dist = []
    max_vars = -1
    max_clauses = -1

    instance_num = 0
    for file in cnf_files:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('p cnf'):
                    tokens = line.strip().split(' ')
                    variables = int(tokens[2])
                    clauses = int(tokens[3])

                    variables_dist.append(variables)
                    clauses_dist.append(clauses)

                    if variables > max_vars:
                        max_vars = variables
                    if clauses > max_clauses:
                        max_clauses = clauses

                    break
        instance_num += 1

    data = zip(variables_dist, clauses_dist)
    return zip(cnf_files, data), max_vars, max_clauses


def print_number_of_instances_per_category(directory, categories):
    res_dict = {}
    for cat in categories:
        instances = collect_instance_names(cat)
        print(cat, len(instances))
        res_dict[cat] = instances
    return res_dict


def generate_satzilla_features(csv_filename):
    data = pd.read_csv(csv_filename)
    idx = 0
    for filename in data['instance_id']:
        idx += 1
        features_filename = filename + '.features'
        if (os.path.exists(features_filename)):
            continue
        print('Creating SATzilla2012 features for file #{0}: {1}...'.format(idx, filename))
        start_process('./third-party/SATzilla2012_features/features', [filename, features_filename])


def filter_zipped_data_by_max_vars_and_clauses(zipped, max_var_limit=None, max_clauses_limit=None):
    if (max_var_limit is None) and (max_clauses_limit is not None):
        raise ValueError('There must be a variable limit if there exists a limit for clauses')
    if max_var_limit is not None:
        if max_clauses_limit is not None:
            filter_fun = lambda t: (t[1][0] < max_var_limit) and (t[1][1] < max_clauses_limit)
        else:
            filter_fun = lambda t: t[1][0] < max_var_limit
        zipped = filter(filter_fun, zipped)
        zipped = list(zipped)
    return zipped


def generate_edgelist_formats(csv_filename, directory='.'):
    data = pd.read_csv(csv_filename)
    idx = 0
    for filename in data['instance_id']:
        filename = os.path.join(os.path.abspath(directory), filename)
        idx += 1
        features_filename = filename + '.edgelist'
        if os.path.exists(features_filename):
            continue
        print('\nCreating Edgelist format for file #{0}: {1}...'.format(idx, filename))
        basedir = os.path.dirname(filename)
        filepath = os.path.basename(filename)
        PARSERSLIB.parse_dimacs_to_edgelist(basedir, filepath)


def generate_dgcnn_formats(csv_filename, csv_labels, directory='.', output_directory="."):
    def log10_transform_data(data):
        minimum_log10_value = 0.001
        data[data < minimum_log10_value] = minimum_log10_value
        return np.log10(data)

    def sort_by_split(column: pd.Series):
        result = []
        for val in column.values:
            if val == "Train":
                result.append(1)
            if val == "Validation":
                result.append(2)
            if val == "Test":
                result.append(3)
            if val == "None":
                result.append(4)
        return pd.Series(result)

    data = pd.read_csv(os.path.join(directory, csv_filename)).sort_values(by="split", key=sort_by_split)
    ys = pd.read_csv(os.path.join(directory, csv_labels))
    features_filenames = []
    instance_ids = []
    splits = {}
    test_ys = []
    for i in range(len(data)):
        split = data.iloc[i]['split']
        if split not in splits:
            splits[split] = 0
        splits[split] += 1

        if split == "None":
            continue

        instance_id = data.iloc[i]['instance_id']
        instance_ids.append(instance_id)
        labels = log10_transform_data(ys[ys["instance_id"] == instance_id].drop(columns="instance_id").values[0])
        if split == "Test":
            test_ys.append(labels)

        filename = os.path.join(os.path.abspath(directory), instance_id)
        features_filename = filename + '.dgcnn.txt'
        features_filenames.append(features_filename)
        if os.path.exists(features_filename):
            continue

        print('\nCreating DGCNN format for file #{0}: {1}...'.format(i + 1, filename))
        basedir = os.path.dirname(filename)
        filepath = os.path.basename(filename)
        PARSERSLIB.parse_dimacs_to_dgcnn_vcg(basedir, filepath, " ".join([str(l) for l in labels]))

    print(f"Found:\n\t{splits['Train']} in Train\n\t{splits['Validation']} in Validation\n\t{splits['Test']} in Test" +
          f"\n\t{splits['None']} in None")

    output_directory = os.path.join(output_directory, "CNF")
    os.makedirs(output_directory, exist_ok=True)

    # Saving true labels
    np.savetxt(os.path.join(output_directory, "test_ytrue.txt"), np.array(test_ys, dtype=np.float32))
    # Saving parsed instance ids
    instance_ids_file = os.path.join(output_directory, "instance_ids.pickled")
    with open(instance_ids_file, "wb") as instance_ids_file:
        pkl.dump([instance_ids, splits], instance_ids_file)
