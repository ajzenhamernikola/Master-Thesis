import os
from datetime import datetime
from multiprocessing import Pool, Lock
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import pandas as pd

from formats.dimacs import Dimacs
from formats.edgelist import Edgelist
from utils.data_conversions import dok_matrix_to_edgelist

import gnn.GNN as GNN
import gnn.gnn_utils

from models import Net

files_to_parse: list = []
file_to_id: dict = {}
file_to_parsed: dict = {}
id_to_file: dict = {}
edge_lists: np.array = np.empty((0, 3), dtype=np.int32)
lock: Lock = None
dimacs_file_size_limit: int = -1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_instances_metadata_that_failed_csv_filename():
    return os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv')


def get_instances_metadata_graph_ids_filename():
    return os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'graph_ids.csv')


def parsing_preparation(cnf_filename: str, raise_error_on_filename_conflict: bool = False):
    edgelist_dirname = os.path.join(os.path.dirname(cnf_filename), '..', 'edgelist')
    if not os.path.exists(edgelist_dirname):
        os.makedirs(edgelist_dirname)

    edgelist_filename = os.path.join(edgelist_dirname, os.path.basename(cnf_filename) + '.edgelist')
    if os.path.exists(edgelist_filename):
        file_to_parsed[cnf_filename] = True
        if raise_error_on_filename_conflict:
            raise ValueError('File ' + edgelist_filename + ' already exists!')
        else:
            return False

    print('Parsing: ' + cnf_filename)
    return edgelist_filename


def parallel_dimacs_to_edgelist_init(l: Lock):
    global lock
    lock = l


def parallel_dimacs_to_edgelist(cnf_filename: str, graph_id: int):
    global lock

    edgelist_filename = parsing_preparation(cnf_filename)
    if not edgelist_filename:
        return

    try:
        dimacs = Dimacs()
        dimacs.load(cnf_filename)

        edgelist = dok_matrix_to_edgelist(dimacs.to_vcg()[0], graph_id)
        edgelist.pickle(edgelist_filename)

        file_to_parsed[cnf_filename] = True
    except MemoryError:
        lock.acquire()
        with open(get_instances_metadata_that_failed_csv_filename(), 'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'MemoryError\n')
        print('MemoryError occured while processing ' + cnf_filename)
        lock.release()
    except OSError:
        lock.acquire()
        with open(get_instances_metadata_that_failed_csv_filename(), 'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'OSError\n')
        print('OSError occured while processing ' + cnf_filename)
        lock.release()


def serial_dimacs_to_edgelist(cnf_filename: str, graph_id: int):
    edgelist_filename = parsing_preparation(cnf_filename)
    if not edgelist_filename:
        return

    try:
        dimacs = Dimacs()
        dimacs.load(cnf_filename)

        edgelist = dok_matrix_to_edgelist(dimacs.to_vcg()[0], graph_id)
        edgelist.pickle(edgelist_filename)

        file_to_parsed[cnf_filename] = True
    except MemoryError:
        with open(get_instances_metadata_that_failed_csv_filename(), 'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'MemoryError\n')
        print('MemoryError occured while processing ' + cnf_filename)
    except OSError:
        with open(get_instances_metadata_that_failed_csv_filename(), 'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'OSError\n')
        print('OSError occured while processing ' + cnf_filename)


def parse():
    global files_to_parse

    if not os.path.exists(get_instances_metadata_that_failed_csv_filename()):
        with open(get_instances_metadata_that_failed_csv_filename(), 'w') as file:
            file.write('instance_name,timestamp,error_type\n')

    print('Choose parallel (p) or serial (s) parsing method:')
    user_input = input()

    if user_input == 'p':
        print('You have chosen parallel parsing!')

        pool_lock = Lock()
        with Pool(processes=8, initializer=parallel_dimacs_to_edgelist_init, initargs=(pool_lock,)) as p:
            p.starmap(parallel_dimacs_to_edgelist, [(files_to_parse[i], i) for i in range(len(files_to_parse))])
    elif user_input == 's':
        print('You have chosen serial parsing!')

        for i in range(len(files_to_parse)):
            serial_dimacs_to_edgelist(files_to_parse[i], i)
    else:
        print('Unknown parsing method! By default, serial parsing method is used!')

        for i in range(len(files_to_parse)):
            serial_dimacs_to_edgelist(files_to_parse[i], i)


def load(ids_to_load: list, labels: list):
    global edge_lists

    filtered_ids = []
    filtered_labels = []

    for i in range(len(ids_to_load)):
        id_to_load = ids_to_load[i]
        filename = os.path.join('..', 'data', 'edgelist', id_to_file[id_to_load] + '.edgelist')
        filename = os.path.abspath(filename)
        print(filename)
        if not os.path.exists(filename):
            continue

        edge_list = Edgelist(id_to_load)
        edge_list.unpickle(filename)
        edge_lists = np.vstack([edge_lists, edge_list.data])

        filtered_ids.append(id_to_load)
        filtered_labels.append(labels[i])

    return filtered_ids, filtered_labels


def main():
    global files_to_parse
    global file_to_id
    global file_to_parsed
    global id_to_file
    global edge_lists
    global dimacs_file_size_limit

    graph_id = 0

    response = int(input('Set the limit (in MB) for DIMACS file size (-1 for no limit):\n'))
    if response > 0:
        dimacs_file_size_limit = response

    for root, _, files in os.walk(os.path.join(os.path.abspath('..'), 'data', 'cnf')):
        for filename in files:
            if filename[-3:] == 'cnf':
                location = os.path.abspath(os.path.join(root, filename))
                file_size = os.stat(location).st_size  # in bytes
                # Only flag file for parsing if its size is under the limits
                if dimacs_file_size_limit != -1 and file_size < dimacs_file_size_limit * 1000000:
                    files_to_parse.append(location)
                # But label all data from the dataset
                file_to_id[filename] = graph_id
                id_to_file[graph_id] = filename
                file_to_parsed[filename] = False
                graph_id += 1

    with open(get_instances_metadata_graph_ids_filename(), 'w') as file:
        file.write('instance_name,graph_id\n')
        for key in file_to_id.keys():
            file.write(key + ',' + str(file_to_id[key]) + '\n')

    # Always try to parse the data
    parse()

    # Save the metadata for which instances are successfully parsed
    with open(os.path.join('..', 'data', 'instances_metadata', 'file_to_parsed.csv'), 'w') as file:
        file.write('instance_name,is_parsed\n')
        for key in file_to_parsed.keys():
            file.write(key + ',' + str(file_to_parsed[key]) + '\n')

    # Analyse the solver data
    graph_ids = pd.read_csv(get_instances_metadata_graph_ids_filename(), index_col=0)
    main_extracted = pd.read_csv(os.path.join('..', 'data', 'instances_metadata', 'main-extracted.csv'), index_col=2)
    main_extracted: pd.DataFrame = pd.concat([graph_ids, main_extracted], axis=1, join='inner')

    solver_counter = main_extracted.groupby('solver').count()['graph_id']
    chosen_solvers = list(solver_counter[solver_counter >= 4].index)

    condition = main_extracted['solver'].transform(lambda value: value in chosen_solvers)
    main_extracted = main_extracted[condition]
    chosen_ids = list(main_extracted['graph_id'])
    labels = list(main_extracted['solver'])

    # Load already parsed data
    chosen_ids, labels = load(chosen_ids, labels)

    # Create the GNN E- and N- matrices
    gnn_e_matrix = edge_lists
    random_node_data = np.array(np.random.random((gnn_e_matrix.shape[0], 1)), dtype=np.int32)
    gnn_n_matrix = np.hstack([random_node_data, gnn_e_matrix[:, -1:]])

    inp, arcnode, graphnode = gnn.gnn_utils.from_EN_to_GNN(gnn_e_matrix, gnn_n_matrix)

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    num_labels = le.transform(labels)
    one_hot_encoded_label_matrix = np.eye(max(num_labels)+1, dtype=np.int32)[num_labels]

    # set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
    threshold = 0.01
    learning_rate = 0.01
    state_dim = 5
    tf.reset_default_graph()
    input_dim = inp.shape[1]
    output_dim = one_hot_encoded_label_matrix.shape[1]
    max_it = 50
    num_epoch = 10000
    optimizer = tf.train.AdamOptimizer

    # initialize state and output network
    net = Net.Net(input_dim, state_dim, output_dim)

    # initialize GNN
    param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
    print(param)

    tensorboard = False

    g = GNN.GNN(net, input_dim, output_dim, state_dim, max_it, optimizer, learning_rate, threshold, graph_based=False,
                param=param, config=config, tensorboard=tensorboard)

    # train the model
    count = 0

    ######

    for j in range(0, num_epoch):
        _, it = g.Train(inputs=inp, ArcNode=arcnode, target=one_hot_encoded_label_matrix, step=count)

        if count % 30 == 0:
            print("Epoch ", count)
            print("Training: ", g.Validate(inp, arcnode, one_hot_encoded_label_matrix, count))

            # end = time.time()
            # print("Epoch {} at time {}".format(j, end-start))
            # start = time.time()

        count = count + 1

    # evaluate on the test set
    # print("\nEvaluate: \n")
    # print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0])[0])
