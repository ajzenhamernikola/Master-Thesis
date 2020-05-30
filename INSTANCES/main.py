import os 

from utils.cnf.instance import collect_instance_names, collect_cnf_files_and_sizes, \
    calculate_numbers_of_variables_and_clauses
from utils.visualisation.lines import plot_zip, plot_zip_two_data

def main():
    this_directory = os.path.dirname(__file__)

    sat12_indu_instances = collect_instance_names(this_directory, 'SAT12-INDU.csv')
    print('INDU:', len(sat12_indu_instances))

    sat12_hand_instances = collect_instance_names(this_directory, 'SAT12-HAND.csv')
    print('HAND:', len(sat12_hand_instances))

    cnf_files, file_sizes = collect_cnf_files_and_sizes(this_directory)
    zipped = zip(cnf_files, file_sizes)
    zipped = sorted(zipped, key=lambda t: t[1])
    file_sizes_figname = os.path.join(this_directory, 'figures', 'filesizes.png')
    plot_zip(zipped, 'Instances', 'Sizes', 'File sizes of CNF instances (in MB)', None, 'discretize', \
        lambda y: y/(1024*1024), file_sizes_figname)

    zipped, max_vars, max_clauses = calculate_numbers_of_variables_and_clauses(cnf_files)
    zipped = sorted(zipped, key=lambda t: t[1][0])
    var_clauses_figname = os.path.join(this_directory, 'figures', 'variables_clauses.png')
    plot_zip_two_data(zipped, 'Instances', ('Variables', 'Clauses'), ('Variable counts', 'Clauses counts'), None, \
        'discretize', None, var_clauses_figname)

    limits = [100, 500, 1000, 1500]
    for limit in limits:
        cnfs = []
        variables = []
        clauses = []
        num = 0
        for t in zipped:
            cnfs.append(t[0])
            variables.append(t[1][0])
            clauses.append(t[1][1])
            num += 1
            if num == limit:
                break
        z = zip(variables, clauses)
        z = zip(cnfs, z)
        var_clauses_figname_limit = os.path.join(this_directory, 'figures', 'variables_clauses_first_' + str(limit) + \
            '.png')
        plot_zip_two_data(z, 'Instances', ('Variables', 'Clauses'), ('Variable counts', 'Clauses counts'), None, \
            'discretize', None, var_clauses_figname_limit)


if __name__ == '__main__':
    main()
