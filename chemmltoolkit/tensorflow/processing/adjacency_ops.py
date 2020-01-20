import tensorflow as tf


def normalise(adjacency_matrices):
    """Normalises adjacency matricies

    This will normalise (scale all rows to sum to one) a set of
    adjacency matricies.

    Args:
        adjacency_matrices: The input tensor of adjacency matricies (can be
        of any rank of at least two, with the adjacency matricies in the last
        two dimensions).

    Returns:
        The resulting tensor of adjacency matricies.
    """
    degree = tf.reduce_sum(adjacency_matrices, axis=-2)
    degree = tf.expand_dims(degree, -1)
    return tf.math.divide_no_nan(adjacency_matrices, degree)


def normalise_spectral(adjacency_matrices):
    """Normalises adjacency matricies with a spectral method

    This will normalise a set of adjacency matricies using the spectral
    method of Kipf and Welling (https://arxiv.org/abs/1609.02907).

    Args:
        adjacency_matrices: The input tensor of adjacency matricies (can be
        of any rank of at least two, with the adjacency matricies in the last
        two dimensions).

    Returns:
        The resulting tensor of adjacency matricies.
    """
    degree = tf.reduce_sum(adjacency_matrices, axis=-2)
    degree_inv = tf.pow(degree, -0.5)
    degree_inv = tf.where(tf.math.is_inf(degree_inv), 0.0, degree_inv)

    # (degree_inv)^T . matrix . (degree_inv)
    di1 = tf.expand_dims(degree_inv, -2)
    di2 = tf.expand_dims(degree_inv, -1)
    return tf.multiply(tf.multiply(di1, adjacency_matrices), di2)
