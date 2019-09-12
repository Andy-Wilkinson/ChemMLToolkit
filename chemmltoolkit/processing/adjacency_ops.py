import numpy as np


def add_master_node(adjacency_matrices):
    """Adds master nodes to adjacency matricies

    This will add a master node (an imaginary node that connects to all other
    nodes) to a set of adjacency matricies. The size of each adjacency matrix
    will hence increase by one in each dimension. Note that the master node
    is not connected to itself.

    Args:
        adjacency_matrices: The input list of adjacency matricies.

    Returns:
        The resulting list of adjacency matricies.
    """
    adj = np.pad(adjacency_matrices,
                 ((0, 0), (0, 1), (0, 1)),
                 mode='constant',
                 constant_values=1)
    adj[:, adj.shape[1]-1, adj.shape[2]-1] = 0
    return adj


def add_self_loops(adjacency_matrices):
    """Adds self loops to adjacency matricies

    This will add a self loops (connections between every node and itself) to
    a set of adjacency matricies.

    Args:
        adjacency_matrices: The input list of adjacency matricies.

    Returns:
        The resulting list of adjacency matricies.
    """
    adj = adjacency_matrices.copy()
    mask = np.eye(adj.shape[1], dtype=bool)
    adj[:, mask] = 1.0
    return adj


def normalise(adjacency_matrices):
    """Normalises adjacency matricies

    This will normalise (scale all rows to sum to one) a set of
    adjacency matricies.

    Args:
        adjacency_matrices: The input list of adjacency matricies.

    Returns:
        The resulting list of adjacency matricies.
    """
    def normalise_matrix(matrix):
        degree = np.sum(matrix, axis=0)
        degree_inv = np.power(degree, -1.0)
        inv_matrix = np.diag(degree_inv)
        return inv_matrix @ matrix

    adjs = [normalise_matrix(matrix) for matrix in adjacency_matrices]
    return np.array(adjs)


def normalise_spectral(adjacency_matrices):
    """Normalises adjacency matricies with a spectral method

    This will normalise a set of adjacency matricies using the spectral
    method of Kipf and Welling (https://arxiv.org/abs/1609.02907).

    Args:
        adjacency_matrices: The input list of adjacency matricies.

    Returns:
        The resulting list of adjacency matricies.
    """
    def normalise_matrix(matrix):
        degree = np.sum(matrix, axis=0)
        degree_inv = np.power(degree, -0.5)
        inv_matrix = np.diag(degree_inv)
        return inv_matrix @ matrix @ inv_matrix

    adjs = [normalise_matrix(matrix) for matrix in adjacency_matrices]
    return np.array(adjs)


def pad(adjacency_matrices, size):
    """Pads adjacency matricies to the desired size

    This will pad the adjacency matricies to the specified size, appending
    zeros as required. The output adjacency matricies will all be of size
    'size' x 'size'.

    Args:
        adjacency_matrices: The input list of adjacency matricies.
        size: The desired dimension of the output matricies.

    Returns:
        The resulting list of adjacency matricies.
    """
    padding = size - adjacency_matrices.shape[1]
    return np.pad(adjacency_matrices,
                  [(0, 0), (0, padding), (0, padding)],
                  mode='constant')
