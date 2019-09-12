import os
from datetime import datetime
from multiprocessing import Pool

from formats.dimacs import Dimacs
from utils.data_conversions import dok_matrix_to_edgelist


def parallel_dimacs_to_edgelist(cnf_filename: str, graph_id: int):
    edgelist_dirname = os.path.join(os.path.dirname(cnf_filename), '..', 'edgelist')
    if not os.path.exists(edgelist_dirname):
        os.makedirs(edgelist_dirname)

    edgelist_filename = os.path.join(edgelist_dirname, os.path.basename(cnf_filename) + '.edgelist')
    if os.path.exists(edgelist_filename):
        return

    print('Parsing: ' + cnf_filename)

    try:
        dimacs = Dimacs()
        dimacs.load(cnf_filename)

        edgelist = dok_matrix_to_edgelist(dimacs.to_vcg()[0], graph_id)
        edgelist.pickle(edgelist_filename)

    except MemoryError:
        with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'), 'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'MemoryError\n')
        print('MemoryError occured while processing ' + cnf_filename)
    except OSError:
        with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'), 'a') as file:
            file.write(cnf_filename + ',' + str(datetime.now()) + ',' + 'OSError\n')
        print('OSError occured while processing ' + cnf_filename)


def main():
    files_to_parse = []
    file_dic = {}
    graph_id = 0

    for root, dirs, files in os.walk(os.path.join(os.path.abspath('..'), 'data', 'cnf')):
        for filename in files:
            if filename[-3:] == 'cnf':
                location = os.path.abspath(os.path.join(root, filename))
                files_to_parse.append(location)
                file_dic[filename] = graph_id
                graph_id += 1

    with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'graph_ids.csv'), 'w') as file:
        file.write('instance_name,graph_id\n')
        for key in file_dic.keys():
            file.write(key + ',' + str(file_dic[key]) + '\n')

    with open(os.path.join(os.path.abspath('..'), 'data', 'instances_metadata', 'instances_that_failed.csv'), 'w') as file:
        file.write('instance_name,timestamp,error_type\n')

    with Pool(processes=8) as p:
        p.starmap(parallel_dimacs_to_edgelist, [(files_to_parse[i], i) for i in range(len(files_to_parse))])

    print('*************************************')
    print('\n\nAll parsing is completed!\n')
    print('*************************************')


if __name__ == '__main__':
    main()
