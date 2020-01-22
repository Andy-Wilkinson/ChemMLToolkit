import numpy as np


def Σ(inputs, function):
    """Sums a function over a range of inputs

    This will call the specified function with a range of input values, and
    sum the results. If the inputs are a tuple then this is treated as a
    range of integers.

    Args:
        inputs: The input values for the function (an iterable, or a tuple
            specifying a range of integers).
        function: A function that takes an input value, returning a result.

    Returns:
        A sum of the results from the function.
    """
    if isinstance(inputs, tuple):
        inputs = range(inputs[0], inputs[1])
    return np.sum([function(i) for i in inputs], axis=0)


def Σneighbours(adjacency_matrix, node, function):
    """Sums a function over all neighbours of a node

    This will call the specified function once for each neighbouring node
    of the specified node. The function will take as input the index of the
    neighbour node.

    Args:
        adjacency_matrix: A single adjacency matrix, with values of 1 for
            connected nodes, and 0 otherwise.
        node: The index of the parent node to find neighbours of.
        function: A function that takes the neighbouring node index, and
        returning a result.

    Returns:
        A sum of the results from the function.
    """
    return np.sum([function(i) for i in range(adjacency_matrix.shape[0])
                   if adjacency_matrix[node, i] == 1], axis=0)


def generate_random_node_features(random,
                                  num_batches,
                                  num_nodes,
                                  num_node_features):
    """Generates a random node feature matrix

    Args:
        random: The NumPy random number generator to use.
        num_batches: The number of batches to generate.
        num_nodes: The number of nodes to generate.
        num_node_features: The number of features per node.

    Returns:
        A NumPy array of shape (num_batches, num_nodes, num_node_features)
        with random values between -1 and 1.
    """
    return random.rand(num_batches, num_nodes, num_node_features) * 2.0 - 1.0


def generate_random_adjacency(random,
                              num_batches,
                              num_nodes,
                              num_edge_features,
                              symmetrise=True,
                              clear_diagonal=True):
    """Generates a random set of adjacency matrices

    Args:
        random: The NumPy random number generator to use.
        num_batches: The number of batches to generate.
        num_nodes: The number of nodes to generate.
        num_edge_features: The number of features per edge.

    Returns:
        A NumPy array of shape (num_batches, num_edge_features, num_nodes,
        num_nodes) with random values of either 0 or 1.
    """
    shape = (num_batches, num_edge_features, num_nodes, num_nodes)
    adjacency = random.randint(0, 2, shape)

    if symmetrise:
        adjacency = np.transpose(adjacency, (0, 1, 3, 2)) * adjacency

    if clear_diagonal:
        mask = np.repeat(np.expand_dims(1 - np.eye(num_nodes), axis=0),
                         num_edge_features, axis=0)
        adjacency = adjacency * mask

    return adjacency
