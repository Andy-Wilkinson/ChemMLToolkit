"""
This type stub file was generated by pyright.
"""

from .dgl_dataset import DGLBuiltinDataset

class KnowledgeGraphDataset(DGLBuiltinDataset):
    """KnowledgeGraph link prediction dataset

    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18

    Parameters
    -----------
    name: str
        Name can be 'FB15k-237', 'FB15k' or 'wn18'.
    reverse: bool
        Whether add reverse edges. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    def __init__(self, name, reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def download(self): # -> None:
        r""" Automatically download data and extract it.
        """
        ...
    
    def process(self): # -> None:
        """
        The original knowledge base is stored in triplets.
        This function will parse these triplets and build the DGLGraph.
        """
        ...
    
    def has_cache(self): # -> bool:
        ...
    
    def __getitem__(self, idx): # -> DGLHeteroGraph:
        ...
    
    def __len__(self): # -> Literal[1]:
        ...
    
    def save(self): # -> None:
        """save the graph list and the labels"""
        ...
    
    def load(self): # -> None:
        ...
    
    @property
    def num_nodes(self): # -> int | Any:
        ...
    
    @property
    def num_rels(self): # -> int | Any:
        ...
    
    @property
    def save_name(self):
        ...
    
    @property
    def train(self): # -> ndarray[Unknown, Unknown]:
        ...
    
    @property
    def valid(self): # -> ndarray[Unknown, Unknown]:
        ...
    
    @property
    def test(self): # -> ndarray[Unknown, Unknown]:
        ...
    


def build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=...):
    """ Create a DGL Homogeneous graph with heterograph info stored as node or edge features.
    """
    ...

class FB15k237Dataset(KnowledgeGraphDataset):
    r"""FB15k237 link prediction dataset.

    .. deprecated:: 0.5.0

        - ``train`` is deprecated, it is replaced by:

            >>> dataset = FB15k237Dataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.edata['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(train_idx)
            >>> rel = graph.edata['etype'][train_idx]

        - ``valid`` is deprecated, it is replaced by:

            >>> dataset = FB15k237Dataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.edata['val_mask']
            >>> val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(val_idx)
            >>> rel = graph.edata['etype'][val_idx]

        - ``test`` is deprecated, it is replaced by:

            >>> dataset = FB15k237Dataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.edata['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(test_idx)
            >>> rel = graph.edata['etype'][test_idx]

    FB15k-237 is a subset of FB15k where inverse
    relations are removed. When creating the dataset,
    a reverse edge with reversed relation types are
    created for each edge by default.

    FB15k237 dataset statistics:

    - Nodes: 14541
    - Number of relation types: 237
    - Number of reversed relation types: 237
    - Label Split:

        - Train: 272115
        - Valid: 17535
        - Test: 20466

    Parameters
    ----------
    reverse : bool
        Whether to add reverse edge. Default True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_nodes: int
        Number of nodes
    num_rels: int
        Number of relation types
    train: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the training graph
    valid: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the validation graph
    test: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the test graph

    Examples
    ----------
    >>> dataset = FB15k237Dataset()
    >>> g = dataset.graph
    >>> e_type = g.edata['e_type']
    >>>
    >>> # get data split
    >>> train_mask = g.edata['train_mask']
    >>> val_mask = g.edata['val_mask']
    >>> test_mask = g.edata['test_mask']
    >>>
    >>> train_set = th.arange(g.number_of_edges())[train_mask]
    >>> val_set = th.arange(g.number_of_edges())[val_mask]
    >>>
    >>> # build train_g
    >>> train_edges = train_set
    >>> train_g = g.edge_subgraph(train_edges,
                                  relabel_nodes=False)
    >>> train_g.edata['e_type'] = e_type[train_edges];
    >>>
    >>> # build val_g
    >>> val_edges = th.cat([train_edges, val_edges])
    >>> val_g = g.edge_subgraph(val_edges,
                                relabel_nodes=False)
    >>> val_g.edata['e_type'] = e_type[val_edges];
    >>>
    >>> # Train, Validation and Test
    """
    def __init__(self, reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx): # -> DGLHeteroGraph:
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, FB15k237Dataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset."""
        ...
    


class FB15kDataset(KnowledgeGraphDataset):
    r"""FB15k link prediction dataset.

    .. deprecated:: 0.5.0

        - ``train`` is deprecated, it is replaced by:

            >>> dataset = FB15kDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.edata['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(train_idx)
            >>> rel = graph.edata['etype'][train_idx]

        - ``valid`` is deprecated, it is replaced by:

            >>> dataset = FB15kDataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.edata['val_mask']
            >>> val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(val_idx)
            >>> rel = graph.edata['etype'][val_idx]

        - ``test`` is deprecated, it is replaced by:

            >>> dataset = FB15kDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.edata['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(test_idx)
            >>> rel = graph.edata['etype'][test_idx]

    The FB15K dataset was introduced in `Translating Embeddings for Modeling
    Multi-relational Data <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_.
    It is a subset of Freebase which contains about
    14,951 entities with 1,345 different relations.
    When creating the dataset, a reverse edge with
    reversed relation types are created for each edge
    by default.

    FB15k dataset statistics:

    - Nodes: 14,951
    - Number of relation types: 1,345
    - Number of reversed relation types: 1,345
    - Label Split:

        - Train: 483142
        - Valid: 50000
        - Test: 59071

    Parameters
    ----------
    reverse : bool
        Whether to add reverse edge. Default True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_nodes: int
        Number of nodes
    num_rels: int
        Number of relation types
    train: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the training graph
    valid: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the validation graph
    test: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the test graph

    Examples
    ----------
    >>> dataset = FB15kDataset()
    >>> g = dataset.graph
    >>> e_type = g.edata['e_type']
    >>>
    >>> # get data split
    >>> train_mask = g.edata['train_mask']
    >>> val_mask = g.edata['val_mask']
    >>>
    >>> train_set = th.arange(g.number_of_edges())[train_mask]
    >>> val_set = th.arange(g.number_of_edges())[val_mask]
    >>>
    >>> # build train_g
    >>> train_edges = train_set
    >>> train_g = g.edge_subgraph(train_edges,
                                  relabel_nodes=False)
    >>> train_g.edata['e_type'] = e_type[train_edges];
    >>>
    >>> # build val_g
    >>> val_edges = th.cat([train_edges, val_edges])
    >>> val_g = g.edge_subgraph(val_edges,
                                relabel_nodes=False)
    >>> val_g.edata['e_type'] = e_type[val_edges];
    >>>
    >>> # Train, Validation and Test
    >>>
    """
    def __init__(self, reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx): # -> DGLHeteroGraph:
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, FB15kDataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset."""
        ...
    


class WN18Dataset(KnowledgeGraphDataset):
    r""" WN18 link prediction dataset.

    .. deprecated:: 0.5.0

        - ``train`` is deprecated, it is replaced by:

            >>> dataset = WN18Dataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.edata['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(train_idx)
            >>> rel = graph.edata['etype'][train_idx]

        - ``valid`` is deprecated, it is replaced by:

            >>> dataset = WN18Dataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.edata['val_mask']
            >>> val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(val_idx)
            >>> rel = graph.edata['etype'][val_idx]

        - ``test`` is deprecated, it is replaced by:

            >>> dataset = WN18Dataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.edata['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(test_idx)
            >>> rel = graph.edata['etype'][test_idx]

    The WN18 dataset was introduced in `Translating Embeddings for Modeling
    Multi-relational Data <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_.
    It included the full 18 relations scraped from
    WordNet for roughly 41,000 synsets. When creating
    the dataset, a reverse edge with reversed relation
    types are created for each edge by default.

    WN18 dataset statistics:

    - Nodes: 40943
    - Number of relation types: 18
    - Number of reversed relation types: 18
    - Label Split:

        - Train: 141442
        - Valid: 5000
        - Test: 5000

    Parameters
    ----------
    reverse : bool
        Whether to add reverse edge. Default True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_nodes: int
        Number of nodes
    num_rels: int
        Number of relation types
    train: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the training graph
    valid: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the validation graph
    test: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the test graph

    Examples
    ----------
    >>> dataset = WN18Dataset()
    >>> g = dataset.graph
    >>> e_type = g.edata['e_type']
    >>>
    >>> # get data split
    >>> train_mask = g.edata['train_mask']
    >>> val_mask = g.edata['val_mask']
    >>>
    >>> train_set = th.arange(g.number_of_edges())[train_mask]
    >>> val_set = th.arange(g.number_of_edges())[val_mask]
    >>>
    >>> # build train_g
    >>> train_edges = train_set
    >>> train_g = g.edge_subgraph(train_edges,
                                  relabel_nodes=False)
    >>> train_g.edata['e_type'] = e_type[train_edges];
    >>>
    >>> # build val_g
    >>> val_edges = th.cat([train_edges, val_edges])
    >>> val_g = g.edge_subgraph(val_edges,
                                relabel_nodes=False)
    >>> val_g.edata['e_type'] = e_type[val_edges];
    >>>
    >>> # Train, Validation and Test
    >>>
    """
    def __init__(self, reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx): # -> DGLHeteroGraph:
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, WN18Dataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset."""
        ...
    


def load_data(dataset): # -> WN18Dataset | FB15kDataset | FB15k237Dataset | None:
    r"""Load knowledge graph dataset for RGCN link prediction tasks

    It supports three datasets: wn18, FB15k and FB15k-237

    Parameters
    ----------
    dataset: str
        The name of the dataset to load.

    Return
    ------
    The dataset object.
    """
    ...

