import os 

from ..visualisation.lines import plot_zip, plot_zip_two_data
from ..cnf.process_cnf_attributes import filter_zipped_data_by_max_vars_and_clauses


def plot_filesizes(directory, zipped):
    file_sizes_figname = os.path.join(directory, 'figures', 'filesizes.png')
    plot_zip(zipped, 'Instances', 'Sizes', 'File sizes of CNF instances (in MB)', None, 'discretize',
             lambda y: y/(1024*1024), file_sizes_figname)


def plot_variables_and_clauses_distrubutions_with_limit(directory, zipped, max_var_limit=None, max_clauses_limit=None):
    zipped = filter_zipped_data_by_max_vars_and_clauses(zipped, max_var_limit, max_clauses_limit)
    figtitle = 'variables_clauses' + (('_max_vars_' + str(max_var_limit)) if max_var_limit is not None else '') + \
        (('_max_clauses_' + str(max_clauses_limit)) if max_clauses_limit is not None else '') + '.png'
    figname = os.path.join(directory, 'figures', figtitle)
    plot_zip_two_data(zipped, 'Instances', ('Variables', 'Clauses'), ('Variable counts', 'Clauses counts'),
                      ('r-', 'bo'), 'discretize', None, figname)


def plot_variables_and_clauses_distrubutions(directory, zipped):
    plot_variables_and_clauses_distrubutions_with_limit(directory, zipped)