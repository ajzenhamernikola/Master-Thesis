import pandas as pd 

from utils.os.path import \
    collect_files_and_sizes
from utils.os.process import \
    start_process
from utils.csv.series import \
    get_column_without_duplicates


def collect_instance_names(csv_filename):
    data = pd.read_csv(csv_filename)
    col = get_column_without_duplicates(data, 'instance_id')
    col.sort()
    return col


def collect_cnf_files_and_sizes(directory):
    return collect_files_and_sizes(directory, '.cnf')


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
    for cat in categories:
        instances = collect_instance_names(cat)
        print(cat, len(instances))


def generate_satzilla_features(csv_filename):
    data = pd.read_csv(csv_filename)
    for filename in data['instance_id']:
        features_filename = filename + '.features'
        start_process('./third-party/SATzilla2012_features/features', [filename, features_filename])