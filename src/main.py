import os
from datetime import datetime
from multiprocessing import Pool, Lock

import numpy as np
import gnn.GNN as GNN
import gnn.gnn_utils
import pandas as pd

from src.formats.dimacs import Dimacs
from src.formats.edgelist import Edgelist
from src.utils.data_conversions import dok_matrix_to_edgelist

files_to_parse: list = []
file_to_id: dict = {}
id_to_file: dict = {}
edge_lists: list = []
lock: Lock = None


def parsing_preparation(cnf_filename: str):
    edgelist_dirname = os.path.join(os.path.dirname(cnf_filename), '..', 'edgelist')
    if not os.path.exists(edgelist_dirname):
        os.makedirs(edgelist_dirname)

    edgelist_filename = os.path.join(edgelist_dirname, os.path.basename(cnf_filename) + '.edgelist')
    if os.path.exists(edgelist_filename):
        return

    print('Parsing: ' + cnf_filename)
    return edgelist_filename


def parallel_dimacs_to_edgelist_init(l: Lock):
    global lock
    lock = l


def parallel_dimacs_to_edgelist(cnf_filename: str, graph_id: int):
    edgelist_filename = parsing_preparation(cnf_filename)

    try:
        dimacs = Dimacs()
        dimacs.load(cnf_filename)

        edgelist = dok_matrix_to_edgelist(dimacs.to_vcg()[0], graph_id)
        edgelist.pickle(edgelist_filename)

    except MemoryError:
        lock.acquire()
        with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'),
                  'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'MemoryError\n')
        print('MemoryError occured while processing ' + cnf_filename)
        lock.release()
    except OSError:
        lock.acquire()
        with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'),
                  'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'OSError\n')
        print('OSError occured while processing ' + cnf_filename)
        lock.release()


def serial_dimacs_to_edgelist(cnf_filename: str, graph_id: int):
    edgelist_filename = parsing_preparation(cnf_filename)

    try:
        dimacs = Dimacs()
        dimacs.load(cnf_filename)

        edgelist = dok_matrix_to_edgelist(dimacs.to_vcg()[0], graph_id)
        edgelist.pickle(edgelist_filename)

    except MemoryError:
        with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'),
                  'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'MemoryError\n')
        print('MemoryError occured while processing ' + cnf_filename)
    except OSError:
        with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'),
                  'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'OSError\n')
        print('OSError occured while processing ' + cnf_filename)


def parse():
    with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'),
              'w') as file:
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


def load(ids_to_load: list = None):
    if ids_to_load is None:
        ids_to_load = np.random.randint(0, len(file_to_id), 50)

    for id_to_load in ids_to_load:
        filename = os.path.join('..', 'data', 'edgelists', id_to_file[id_to_load] + '.edgelist')
        if not os.path.exists(filename):
            raise ValueError(filename + ' does not exist!')

        edge_list = Edgelist(id_to_load)
        edge_list.unpickle(filename)
        edge_lists.append(edge_list)


def main():
    global files_to_parse
    global file_to_id
    global id_to_file
    global edge_lists

    graph_id = 0

    for root, _, files in os.walk(os.path.join(os.path.abspath('..'), 'data', 'cnf')):
        for filename in files:
            if filename[-3:] == 'cnf':
                location = os.path.abspath(os.path.join(root, filename))
                files_to_parse.append(location)
                file_to_id[filename] = graph_id
                id_to_file[graph_id] = filename
                graph_id += 1

    with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'graph_ids.csv'), 'w') as file:
        file.write('instance_name,graph_id\n')
        for key in file_to_id.keys():
            file.write(key + ',' + str(file_to_id[key]) + '\n')

    response = input('Parse the data? [y/n]\n')
    if response == 'y':
        parse()

    graph_ids = pd.read_csv(os.path.join('..', 'data', 'instances_metadata', 'graph_ids.csv'), index_col=0)
    main_extracted = pd.read_csv(os.path.join('..', 'data', 'instances_metadata', 'main-extracted.csv'), index_col=2)
    main_extracted: pd.DataFrame = pd.concat([graph_ids, main_extracted], axis=1, join='inner')

    solver_counter = main_extracted.groupby('solver').count()['graph_id']
    chosen_solvers = list(solver_counter[solver_counter >= 4].index)

    condition = main_extracted['solver'].transform(lambda value: value in chosen_solvers)
    main_extracted = main_extracted[condition]
    chosen_ids = list(main_extracted['graph_id'])
    labels = list(main_extracted['solver'])

    load(chosen_ids)

    gnn_e_matrix = []
    for edge_list in edge_lists:
        gnn_e_matrix += edge_list


if __name__ == '__main__':
    main()
