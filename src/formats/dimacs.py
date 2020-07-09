import re

import numpy
from scipy import sparse


class DimacsNotLoadedError(Exception):
    def __init__(self, message):
        self.message = message


class DimacsBadlyFormattedError(Exception):
    def __init__(self, message):
        self.message = message


class Dimacs:
    def __init__(self):
        self.num_of_vars = -1
        self.num_of_clauses = -1
        self.parsed_from_file = False
        self.clauses = []
        self.__vcg_is_formed__ = False
        self.__vcg_matrix__ = None
        self.__vcg_dim__ = None
        self.__vg_is_formed__ = False
        self.__vg_matrix__ = None
        self.__vg_dim__ = None
        self.__cg_is_formed__ = False
        self.__cg_matrix__ = None
        self.__cg_dim__ = None

    def load(self, location):
        with open(location) as f:
            self.__load__(f)

    def __load__(self, file_content):
        self.num_of_vars = -1
        self.num_of_clauses = -1
        self.clauses = []
        assert_completed = False

        p_comment = re.compile(r'c.*')
        p_stats = re.compile(r'p\s*cnf\s*(\d*)\s*(\d*)')

        for line in file_content:
            # Only deal with lines that aren't comments
            if not p_comment.match(line):
                m = p_stats.match(line)

                if not m:
                    if not assert_completed:
                        if self.num_of_vars == -1:
                            raise DimacsBadlyFormattedError('Unknown number of variables before parsing clauses')
                        if self.num_of_clauses == -1:
                            raise DimacsBadlyFormattedError('Unknown number of clauses before parsing clauses')
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
                                list_num.append(num + self.num_of_vars // 2)

                    if len(list_num) > 0:
                        self.clauses.append(list_num)

                else:
                    self.num_of_vars = 2 * int(m.group(1))
                    self.num_of_clauses = int(m.group(2))

        self.parsed_from_file = True

    # Variable-Clause Graph (VCG) is a bipartite graph with a node for each variable, a node
    # for each clause, and an edge between them whenever a variable occurs in a clause.
    def to_vcg(self):
        if not self.parsed_from_file:
            raise DimacsNotLoadedError('Trying to get the variable clause graph from unknown dimacs format')

        if self.__vcg_is_formed__:
            return self.__vcg_matrix__, self.__vcg_dim__

        dim = self.num_of_vars + self.num_of_clauses
        matrix = sparse.dok_matrix((dim, dim), dtype=numpy.int32)

        for c_ind in range(len(self.clauses)):
            for lit in self.clauses[c_ind]:
                matrix[lit - 1, c_ind + self.num_of_vars] = 1
                matrix[c_ind + self.num_of_vars, lit - 1] = 1

        self.__vcg_matrix__ = matrix
        self.__vcg_dim__ = dim
        self.__vcg_is_formed__ = True

        return self.__vcg_matrix__, self.__vcg_dim__

    # Variable Graph (VG) has a node for each variable and an edge between variables that occur
    # together in at least one clause.
    def to_vg(self):
        if not self.parsed_from_file:
            raise DimacsNotLoadedError('Trying to get the variable clause graph from unknown dimacs format')

        if self.__vg_is_formed__:
            return self.__vg_matrix__, self.__vg_dim__

        dim = self.num_of_vars
        matrix = sparse.dok_matrix((dim, dim), dtype=numpy.int32)

        for i in range(len(self.clauses)):
            c = self.clauses[i]
            c_len = len(c)

            for j in range(c_len):
                for k in range(j + 1, c_len):
                    matrix[c[j] - 1, c[k] - 1] = 1
                    matrix[c[k] - 1, c[j] - 1] = 1

        self.__vg_matrix__ = matrix
        self.__vg_dim__ = dim
        self.__vg_is_formed__ = True

        return self.__vg_matrix__, self.__vg_dim__

    # Clause Graph (CG) has nodes representing clauses and
    # an edge between two clauses whenever they share a negated literal.
    def to_cg(self):
        if not self.parsed_from_file:
            raise DimacsNotLoadedError('Trying to get the variable clause graph from unknown dimacs format')

        if self.__cg_is_formed__:
            return self.__cg_matrix__, self.__cg_dim__

        dim = self.num_of_clauses
        matrix = sparse.dok_matrix((dim, dim), dtype=numpy.int32)

        for i in range(dim):
            for j in range(i + 1, dim):
                c_i = self.clauses[i]
                c_j = self.clauses[j]

                for lit_c_i in c_i:
                    if lit_c_i >= self.num_of_vars // 2 and lit_c_i in c_j:
                        matrix[i, j] = 1
                        matrix[j, i] = 1

        self.__cg_matrix__ = matrix
        self.__cg_dim__ = dim
        self.__cg_is_formed__ = True

        return self.__cg_matrix__, self.__cg_dim__
