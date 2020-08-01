import os
import pandas as pd
import pickle as pkl
import numpy as np

from ..parsers.libparsers import parserslib
from ..os.process import start_process
from ..cnf.process_cnf_attributes import sort_by_split, is_unsolvable
from ..algorithms.math import log10_transform_data


def generate_satzilla_features(csv_filename):
    data = pd.read_csv(csv_filename)
    idx = 0
    for filename in data['instance_id']:
        idx += 1
        features_filename = filename + '.features'
        if (os.path.exists(features_filename)):
            continue
        print('Creating SATzilla2012 features for file #{0}: {1}...'.format(idx, filename))
        root_dirname = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        start_process(f'{root_dirname}/third-party/SATzilla2012_features/features', [filename, features_filename])


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
        parserslib.parse_dimacs_to_edgelist(basedir, filepath)


def generate_dgcnn_formats(csv_filename, csv_labels, directory='.', output_directory="."):
    all_data_y = pd.read_csv(os.path.join(directory, "chosen_data", "all_data_y.csv"))

    data = pd.read_csv(os.path.join(directory, csv_filename)).sort_values(by="split", key=sort_by_split)
    ys = pd.read_csv(os.path.join(directory, csv_labels))
    features_filenames = []
    instance_ids = []
    splits = {}
    test_ys = []
    for i in range(len(data)):
        instance_id = data.iloc[i]['instance_id']
        if is_unsolvable(all_data_y, instance_id):
            continue

        split = data.iloc[i]['split']
        if split not in splits:
            splits[split] = 0
        splits[split] += 1

        if split == "None":
            continue

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
        parserslib.parse_dimacs_to_dgcnn_vcg(basedir, filepath, " ".join([str(l) for l in labels]))

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
