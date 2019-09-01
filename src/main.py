from __future__ import absolute_import

import os
from multiprocessing import Pool

from src.models.dimacs import dimacs_to_matrix
from src.utils.data_conversions import dok_matrix_to_edgelist


def parallel_parse_file(location: str):
    if os.path.exists(location + '.edgelist'):
        return

    print('Parsing ' + location)

    try:
        matrices, dims, times = dimacs_to_matrix(location, 1)
        for i in range(len(matrices)):
            dir_name = os.path.join(os.path.abspath('..'), 'test')
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            dok_matrix_to_edgelist(matrices[i], location)

    except MemoryError:
        print('*************************************')
        print('MemoryError occured on data: ' + location)
        print('*************************************')
    except OSError:
        print('*************************************')
        print('OSError occured on data: ' + location)
        print('*************************************')


def main():
    files_to_parse = []
    file_dic = {}
    graph_id = 0

    for root, dirs, files in os.walk(os.path.join(os.path.abspath('..'), 'instances')):
        for filename in files:
            if filename[-3:] == 'cnf':
                location = os.path.join(root, filename)
                files_to_parse.append(location)
                file_dic[filename] = graph_id
                graph_id += 1

    with open(os.path.join(os.path.abspath('..'), 'instances', 'test', 'graph_ids.csv'), 'w') as file:
        for key in file_dic.keys():
            file.write(key + ',' + str(file_dic[key]) + '\n')

    with Pool(processes=10) as p:
        p.map(parallel_parse_file, files_to_parse)

    print('*************************************')
    print('\n\nAll parsing is completed!\n')
    print('*************************************')


if __name__ == '__main__':
    main()
