"""
This type stub file was generated by pyright.
"""

"""PinSAGE sampler & related functions and classes"""
class RandomWalkNeighborSampler:
    """PinSage-like neighbor sampler extended to any heterogeneous graphs.

    Given a heterogeneous graph and a list of nodes, this callable will generate a homogeneous
    graph where the neighbors of each given node are the most commonly visited nodes of the
    same type by multiple random walks starting from that given node.  Each random walk consists
    of multiple metapath-based traversals, with a probability of termination after each traversal.

    The edges of the returned homogeneous graph will connect to the given nodes from their most
    commonly visited nodes, with a feature indicating the number of visits.

    The metapath must have the same beginning and ending node type to make the algorithm work.

    This is a generalization of PinSAGE sampler which only works on bidirectional bipartite
    graphs.

    Parameters
    ----------
    G : DGLGraph
        The graph.  It must be on CPU.
    num_traversals : int
        The maximum number of metapath-based traversals for a single random walk.

        Usually considered a hyperparameter.
    termination_prob : float
        Termination probability after each metapath-based traversal.

        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each given node.

        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors (or most commonly visited nodes) to select for each given node.
    metapath : list[str] or list[tuple[str, str, str]], optional
        The metapath.

        If not given, DGL assumes that the graph is homogeneous and the metapath consists
        of one step over the single edge type.
    weight_column : str, default "weights"
        The name of the edge feature to be stored on the returned graph with the number of
        visits.

    Examples
    --------
    See examples in :any:`PinSAGESampler`.
    """
    def __init__(self, G, num_traversals, termination_prob, num_random_walks, num_neighbors, metapath=..., weight_column=...) -> None:
        ...
    
    def __call__(self, seed_nodes): # -> DGLHeteroGraph:
        """
        Parameters
        ----------
        seed_nodes : Tensor
            A tensor of given node IDs of node type ``ntype`` to generate neighbors from.  The
            node type ``ntype`` is the beginning and ending node type of the given metapath.

            It must be on CPU and have the same dtype as the ID type of the graph.

        Returns
        -------
        g : DGLGraph
            A homogeneous graph constructed by selecting neighbors for each given node according
            to the algorithm above.  The returned graph is on CPU.
        """
        ...
    


class PinSAGESampler(RandomWalkNeighborSampler):
    """PinSAGE-like neighbor sampler.

    This callable works on a bidirectional bipartite graph with edge types
    ``(ntype, fwtype, other_type)`` and ``(other_type, bwtype, ntype)`` (where ``ntype``,
    ``fwtype``, ``bwtype`` and ``other_type`` could be arbitrary type names).  It will generate
    a homogeneous graph of node type ``ntype`` where the neighbors of each given node are the
    most commonly visited nodes of the same type by multiple random walks starting from that
    given node.  Each random walk consists of multiple metapath-based traversals, with a
    probability of termination after each traversal.  The metapath is always ``[fwtype, bwtype]``,
    walking from node type ``ntype`` to node type ``other_type`` then back to ``ntype``.

    The edges of the returned homogeneous graph will connect to the given nodes from their most
    commonly visited nodes, with a feature indicating the number of visits.

    Parameters
    ----------
    G : DGLGraph
        The bidirectional bipartite graph.

        The graph should only have two node types: ``ntype`` and ``other_type``.
        The graph should only have two edge types, one connecting from ``ntype`` to
        ``other_type``, and another connecting from ``other_type`` to ``ntype``.
    ntype : str
        The node type for which the graph would be constructed on.
    other_type : str
        The other node type.
    num_traversals : int
        The maximum number of metapath-based traversals for a single random walk.

        Usually considered a hyperparameter.
    termination_prob : int
        Termination probability after each metapath-based traversal.

        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each given node.

        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors (or most commonly visited nodes) to select for each given node.
    weight_column : str, default "weights"
        The name of the edge feature to be stored on the returned graph with the number of
        visits.

    Examples
    --------
    Generate a random bidirectional bipartite graph with 3000 "A" nodes and 5000 "B" nodes.

    >>> g = scipy.sparse.random(3000, 5000, 0.003)
    >>> G = dgl.heterograph({
    ...     ('A', 'AB', 'B'): g.nonzero(),
    ...     ('B', 'BA', 'A'): g.T.nonzero()})

    Then we create a PinSage neighbor sampler that samples a graph of node type "A".  Each
    node would have (a maximum of) 10 neighbors.

    >>> sampler = dgl.sampling.PinSAGESampler(G, 'A', 'B', 3, 0.5, 200, 10)

    This is how we select the neighbors for node #0, #1 and #2 of type "A" according to
    PinSAGE algorithm:

    >>> seeds = torch.LongTensor([0, 1, 2])
    >>> frontier = sampler(seeds)
    >>> frontier.all_edges(form='uv')
    (tensor([ 230,    0,  802,   47,   50, 1639, 1533,  406, 2110, 2687, 2408, 2823,
                0,  972, 1230, 1658, 2373, 1289, 1745, 2918, 1818, 1951, 1191, 1089,
             1282,  566, 2541, 1505, 1022,  812]),
     tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2]))

    For an end-to-end example of PinSAGE model, including sampling on multiple layers
    and computing with the sampled graphs, please refer to our PinSage example
    in ``examples/pytorch/pinsage``.

    References
    ----------
    Graph Convolutional Neural Networks for Web-Scale Recommender Systems
        Ying et al., 2018, https://arxiv.org/abs/1806.01973
    """
    def __init__(self, G, ntype, other_type, num_traversals, termination_prob, num_random_walks, num_neighbors, weight_column=...) -> None:
        ...
    


