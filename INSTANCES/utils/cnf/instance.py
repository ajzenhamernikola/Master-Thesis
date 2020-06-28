import os
import pandas as pd

from utils.os.path import \
    collect_files_and_sizes
from utils.os.process import \
    start_process
from utils.csv.series import \
    get_column_without_duplicates
from utils.parsers.libparsers import \
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


def generate_node2vec_features(csv_filename, directory='.'):
    data = pd.read_csv(csv_filename)
    idx = 0
    for filename in data['instance_id']:
        filename = os.path.join(os.path.abspath(directory), filename)
        idx += 1
        edgelist_filename = filename + '.edgelist'
        if not os.path.exists(edgelist_filename):
            raise FileNotFoundError(f"Could not find edgelist file: {edgelist_filename}")
        emb_filename = filename + '.emb'
        if os.path.exists(emb_filename):
            continue
        print('\nCreating Node2Vec format for file #{0}: {1}...'.format(idx, edgelist_filename))
        args = [
            '-i:{0}'.format(edgelist_filename), '-o:{0}'.format(emb_filename), '-l:3', '-d:2', '-r:5', '-p:0.3', '-dr',
            '-v'
        ]
        print(f"Calling ./third-party/node2vec/node2vec {' '.join(args)}")
        start_process('./third-party/node2vec/node2vec', args)
