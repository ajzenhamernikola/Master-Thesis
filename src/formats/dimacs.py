import re
import os
import time

import numpy
from scipy import sparse
from matplotlib import pyplot


def load(file_content):
    clauses = []
    num_of_vars = -1
    num_of_clauses = -1
    assert_completed = False

    p_comment = re.compile(r'c.*')
    p_stats = re.compile(r'p\s*cnf\s*(\d*)\s*(\d*)')

    for line in file_content:
        # Only deal with lines that aren't comments
        if not p_comment.match(line):
            m = p_stats.match(line)

            if not m:
                if not assert_completed:
                    if num_of_vars == -1 or num_of_clauses == -1:
                        raise ValueError('One of the values num_of_vars or num_of_clauses is missing')
                    assert_completed = True

                nums = line.rstrip('\n').split(' ')
                list_num = []
                for lit in nums:
                    if lit != '':
                        if int(lit) == 0:
                            continue
                        num = abs(int(lit))
                        sign = True
                        if int(lit) < 0:
                            sign = False

                        if sign:
                            list_num.append(num)
                        else:
                            list_num.append(num + num_of_vars // 2)

                if len(list_num) > 0:
                    clauses.append(list_num)

            else:
                num_of_vars = 2 * int(m.group(1))
                num_of_clauses = int(m.group(2))

    return clauses, num_of_vars, num_of_clauses


def load_file(location):
    with open(location) as f:
        return load(f)


# Variable-Clause Graph (VCG) is a bipartite graph with a node for each variable, a node
# for each clause, and an edge between them whenever a variable occurs in a clause.
def vcg(location):
    clauses, num_of_vars, num_of_clauses = load_file(location)
    dim = num_of_vars + num_of_clauses
    matrix = sparse.dok_matrix((dim, dim), dtype=numpy.int32)

    for c_ind in range(len(clauses)):
        for lit in clauses[c_ind]:
            matrix[lit - 1, c_ind + num_of_vars] = 1
            matrix[c_ind + num_of_vars, lit - 1] = 1

    return matrix, dim


# Variable Graph (VG) has a node for each variable and an edge between variables that occur
# together in at least one clause.
def vg(location):
    clauses, num_of_vars, _ = load_file(location)
    dim = num_of_vars
    matrix = sparse.dok_matrix((dim, dim), dtype=numpy.int32)

    for i in range(len(clauses)):
        c = clauses[i]
        c_len = len(c)

        for j in range(c_len):
            for k in range(j + 1, c_len):
                matrix[c[j] - 1, c[k] - 1] = 1
                matrix[c[k] - 1, c[j] - 1] = 1

    return matrix, dim


# Clause Graph (CG) has nodes representing clauses and
# an edge between two clauses whenever they share a negated literal.
def cg(location):
    clauses, num_of_vars, num_of_clauses = load_file(location)
    dim = num_of_clauses
    matrix = sparse.dok_matrix((dim, dim), dtype=numpy.int32)

    for i in range(dim):
        for j in range(i + 1, dim):
            c_i = clauses[i]
            c_j = clauses[j]

            for lit_c_i in c_i:
                if lit_c_i >= num_of_vars // 2 and lit_c_i in c_j:
                    matrix[i, j] = 1
                    matrix[j, i] = 1

    return matrix, dim


def dimacs_to_matrix(location: str, number_of_graphs: int = -1):
    all_functions = [vcg, vg, cg]

    if number_of_graphs < 1 and number_of_graphs != -1:
        raise ValueError('Value num_of_graphs must be 1 or greater')
    elif number_of_graphs == -1:
        number_of_graphs = len(all_functions)

    functions = all_functions[0: number_of_graphs]
    num_of_functions = len(functions)
    matrices = []
    dims = []
    times = []

    for i in range(num_of_functions):
        f = functions[i]

        time_begin = time.clock()
        matrix, dim = f(location)
        time_end = time.clock()

        time_diff = round(time_end - time_begin, 2)
        # print('Created an adjacency matrix for {} of dim {} x {} in {} seconds'.format(f.__name__.upper(), dim, dim,
        #                                                                                time_diff))

        # create_figure_from_matrix(x_matrix, dim, f, i, '../test')

        matrices.append(matrix)
        dims.append(dim)
        times.append(time_diff)

    return matrices, dims, times


def create_figure_from_matrix(x_matrix, dim, f, i, dirname):
    print('Number of nonzero elements in matrix {} of size {} is {}'.format(
        f.__name__.upper(),
        dim,
        x_matrix.nnz))
    pyplot.axis("off")
    fig = pyplot.imshow(x_matrix.todense())
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    pyplot.savefig(os.path.join(dirname, '{}_{}.png'.format(i, f.__name__.upper())),
                   bbox_inches='tight', pad_inches=0, dpi=300)
    pyplot.close()
