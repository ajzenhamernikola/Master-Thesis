import os
import pandas as pd
import numpy as np

from ..os.path import collect_files_and_sizes
from ..algorithms.series import get_column_without_duplicates, sort_by_array_order


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


def save_cnf_zipped_data_to_csv(data, filename, max_var_limit=None, max_clauses_limit=None):
    data = filter_zipped_data_by_max_vars_and_clauses(data, max_var_limit, max_clauses_limit)
    if len(data) == 0:
        raise ValueError('No data to save')

    df = []
    for item in data:
        instance_id = item[0]
        variables = item[1][0]
        clauses = item[1][1]
        df.append([instance_id, variables, clauses])

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df = pd.DataFrame(df, columns=['instance_id', 'variables', 'clauses'])
    print('Saving {} instances to file: {}'.format(len(df), filename))
    df.to_csv(filename, index=False)


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


def sort_by_split(column: pd.Series):
    return sort_by_array_order(column, ["Train", "Validation", "Test", "None"])


def is_unsolvable(data: pd.DataFrame, instance_id: str):
    row = data[data["instance_id"] == instance_id].drop(columns="instance_id").values
    return np.all(row == 1200.0)
