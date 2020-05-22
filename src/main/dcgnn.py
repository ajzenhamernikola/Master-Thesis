import os 
import pandas as pd
from sklearn import preprocessing

from parsers.libparsers import PARSERSLIB


def get_metadata_main_extracted_filename():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, '../..', 'data/metadata/main-extracted.csv')


def get_cnf_directory():
    return '/media/nikola/Elements/Git/Master-Thesis/data/cnf'


def get_parsed_cnf_directory():
    return os.path.join(get_cnf_directory(), 'parsed')


def get_all_data_filename():
    return os.path.join(get_parsed_cnf_directory(), 'all_data.txt')


def main():
    if not os.path.exists(get_parsed_cnf_directory()):
        main_extracted = pd.read_csv(get_metadata_main_extracted_filename())
        labels = main_extracted['solver']
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        num_labels = le.transform(labels)
        main_extracted['solver'] = num_labels
        for x in range(len(main_extracted['benchmark name'])):
            file_name = main_extracted['benchmark name'][x]
            label = main_extracted['solver'][x]
            PARSERSLIB.parse_dimacs_to_dcgnn_vcg(get_cnf_directory(), file_name, label)

    number_of_graphs = 0
    for _, _, files in os.walk(get_parsed_cnf_directory()):
        for _ in files:
            number_of_graphs += 1

    with open(get_all_data_filename(), 'w') as all_data:
        all_data.write(str(number_of_graphs))
        all_data.write('\n')
        for subdir, _, files in os.walk(get_parsed_cnf_directory()):
            for filename in files:
                file_path = os.path.join(subdir, filename)
                print('Adding graph file:', file_path)
                with open(file_path, 'r') as graph_file:
                    all_data.write(graph_file.read())
    