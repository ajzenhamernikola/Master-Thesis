from scipy import sparse


def dok_matrix_to_edgelist(matrix: sparse.dok_matrix, filename: str):
    with open(filename + '.edgelist', 'w') as file:
        for key in matrix.keys():
            file.write(str(key[0]) + ' ' + str(key[1]) + '\n')
