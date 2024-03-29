"""
This type stub file was generated by pyright.
"""

from .dgl_dataset import DGLDataset

"""Dataset for stochastic block model."""
def sbm(n_blocks, block_size, p, q, rng=...):
    """ (Symmetric) Stochastic Block Model

    Parameters
    ----------
    n_blocks : int
        Number of blocks.
    block_size : int
        Block size.
    p : float
        Probability for intra-community edge.
    q : float
        Probability for inter-community edge.
    rng : numpy.random.RandomState, optional
        Random number generator.

    Returns
    -------
    scipy sparse matrix
        The adjacency matrix of generated graph.
    """
    ...

class SBMMixtureDataset(DGLDataset):
    r""" Symmetric Stochastic Block Model Mixture

    Reference: Appendix C of `Supervised Community Detection with Hierarchical Graph Neural Networks <https://arxiv.org/abs/1705.08415>`_

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    n_nodes : int
        Number of nodes.
    n_communities : int
        Number of communities.
    k : int, optional
        Multiplier. Default: 2
    avg_deg : int, optional
        Average degree. Default: 3
    pq : list of pair of nonnegative float or str, optional
        Random densities. This parameter is for future extension,
        for now it's always using the default value.
        Default: Appendix_C
    rng : numpy.random.RandomState, optional
        Random number generator. If not given, it's numpy.random.RandomState() with `seed=None`,
        which read data from /dev/urandom (or the Windows analogue) if available or seed from
        the clock otherwise.
        Default: None

    Raises
    ------
    RuntimeError is raised if pq is not a list or string.

    Examples
    --------
    >>> data = SBMMixtureDataset(n_graphs=16, n_nodes=10000, n_communities=2)
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(data, batch_size=1, collate_fn=data.collate_fn)
    >>> for graph, line_graph, graph_degrees, line_graph_degrees, pm_pd in dataloader:
    ...     # your code here
    """
    def __init__(self, n_graphs, n_nodes, n_communities, k=..., avg_deg=..., pq=..., rng=...) -> None:
        ...
    
    def process(self): # -> None:
        ...
    
    def has_cache(self): # -> bool:
        ...
    
    def save(self): # -> None:
        ...
    
    def load(self): # -> None:
        ...
    
    def __len__(self): # -> int:
        r"""Number of graphs in the dataset."""
        ...
    
    def __getitem__(self, idx): # -> tuple[DGLHeteroGraph | Unknown, Unknown, Unknown | Any, Unknown | Any, Unknown | Any]:
        r""" Get one example by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        graph: :class:`dgl.DGLGraph`
            The original graph
        line_graph: :class:`dgl.DGLGraph`
            The line graph of `graph`
        graph_degree: numpy.ndarray
            In degrees for each node in `graph`
        line_graph_degree: numpy.ndarray
            In degrees for each node in `line_graph`
        pm_pd: numpy.ndarray
            Edge indicator matrices Pm and Pd
        """
        ...
    
    def collate_fn(self, x): # -> tuple[Any, Any, Unknown, Unknown, Unknown]:
        r""" The `collate` function for dataloader

        Parameters
        ----------
        x : tuple
            a batch of data that contains:

            - graph: :class:`dgl.DGLGraph`
                The original graph
            - line_graph: :class:`dgl.DGLGraph`
                The line graph of `graph`
            - graph_degree: numpy.ndarray
                In degrees for each node in `graph`
            - line_graph_degree: numpy.ndarray
                In degrees for each node in `line_graph`
            - pm_pd: numpy.ndarray
                Edge indicator matrices Pm and Pd

        Returns
        -------
        g_batch: :class:`dgl.DGLGraph`
            Batched graphs
        lg_batch: :class:`dgl.DGLGraph`
            Batched line graphs
        degg_batch: numpy.ndarray
            A batch of in degrees for each node in `g_batch`
        deglg_batch: numpy.ndarray
            A batch of in degrees for each node in `lg_batch`
        pm_pd_batch: numpy.ndarray
            A batch of edge indicator matrices Pm and Pd
        """
        ...
    


SBMMixture = SBMMixtureDataset
