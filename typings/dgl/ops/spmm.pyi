"""
This type stub file was generated by pyright.
"""

"""dgl spmm operator module."""
__all__ = ['gspmm']
def reshape_lhs_rhs(lhs_data, rhs_data): # -> tuple[Unknown, Unknown]:
    r""" Expand dims so that there will be no broadcasting issues with different
    number of dimensions. For example, given two shapes (N, 3, 1), (E, 5, 3, 4)
    that are valid broadcastable shapes, change them to (N, 1, 3, 1) and
    (E, 5, 3, 4)

    Parameters
    ----------
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    """
    ...

def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    r""" Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel.

    1. Computes messages by :attr:`op` source node and edge features.
    2. Aggregate the messages by :attr:`reduce_op` as the features on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math:`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``,
        ``copy_lhs``, ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``, ``mean``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by the op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by the op.

    Returns
    -------
    tensor
        The result tensor.
    """
    ...

