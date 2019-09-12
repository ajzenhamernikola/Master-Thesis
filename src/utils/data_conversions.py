from scipy import sparse

from formats.edgelist import Edgelist


def dok_matrix_to_edgelist(matrix: sparse.dok_matrix, graph_id: int):
    edgelist = Edgelist(graph_id)
    for key in matrix.keys():
        edgelist.add_edge(key[0], key[1])

    return edgelist
