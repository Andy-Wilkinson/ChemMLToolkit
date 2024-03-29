"""
This type stub file was generated by pyright.
"""

from .dgl_dataset import DGLBuiltinDataset

"""GNN Benchmark datasets for node classification."""
__all__ = ["AmazonCoBuyComputerDataset", "AmazonCoBuyPhotoDataset", "CoauthorPhysicsDataset", "CoauthorCSDataset", "CoraFullDataset", "AmazonCoBuy", "Coauthor", "CoraFull"]
def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    ...

class GNNBenchmarkDataset(DGLBuiltinDataset):
    r"""Base Class for GNN Benchmark dataset

    Reference: https://github.com/shchur/gnn-benchmark#datasets
    """
    def __init__(self, name, raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def process(self): # -> None:
        ...
    
    def has_cache(self): # -> bool:
        ...
    
    def save(self): # -> None:
        ...
    
    def load(self): # -> None:
        ...
    
    @property
    def num_classes(self):
        """Number of classes."""
        ...
    
    @property
    def data(self): # -> list[DGLHeteroGraph] | list[Unknown]:
        ...
    
    def __getitem__(self, idx): # -> DGLHeteroGraph:
        r""" Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""Number of graphs in the dataset"""
        ...
    


class CoraFullDataset(GNNBenchmarkDataset):
    r"""CORA-Full dataset for node classification task.

    .. deprecated:: 0.5.0

        - ``data`` is deprecated, it is repalced by:

        >>> dataset = CoraFullDataset()
        >>> graph = dataset[0]

    Extended Cora dataset. Nodes represent paper and edges represent citations.

    Reference: `<https://github.com/shchur/gnn-benchmark#datasets>`_

    Statistics:

    - Nodes: 19,793
    - Edges: 126,842 (note that the original dataset has 65,311 edges but DGL adds
      the reverse edges and remove the duplicates, hence with a different number)
    - Number of Classes: 70
    - Node feature size: 8,710

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node.
    data : list
        A list of DGLGraph objects

    Examples
    --------
    >>> data = CoraFullDataset()
    >>> g = data[0]
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    @property
    def num_classes(self): # -> Literal[70]:
        """Number of classes.

        Return
        -------
        int
        """
        ...
    


class CoauthorCSDataset(GNNBenchmarkDataset):
    r""" 'Computer Science (CS)' part of the Coauthor dataset for node classification task.

    .. deprecated:: 0.5.0

        - ``data`` is deprecated, it is repalced by:

        >>> dataset = CoauthorCSDataset()
        >>> graph = dataset[0]

    Coauthor CS and Coauthor Physics are co-authorship graphs based on the Microsoft Academic Graph
    from the KDD Cup 2016 challenge. Here, nodes are authors, that are connected by an edge if they
    co-authored a paper; node features represent paper keywords for each author’s papers, and class
    labels indicate most active fields of study for each author.

    Reference: `<https://github.com/shchur/gnn-benchmark#datasets>`_

    Statistics:

    - Nodes: 18,333
    - Edges: 163,788 (note that the original dataset has 81,894 edges but DGL adds
      the reverse edges and remove the duplicates, hence with a different number)
    - Number of classes: 15
    - Node feature size: 6,805

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node.
    data : list
        A list of DGLGraph objects

    Examples
    --------
    >>> data = CoauthorCSDataset()
    >>> g = data[0]
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    @property
    def num_classes(self): # -> Literal[15]:
        """Number of classes.

        Return
        -------
        int
        """
        ...
    


class CoauthorPhysicsDataset(GNNBenchmarkDataset):
    r""" 'Physics' part of the Coauthor dataset for node classification task.

    .. deprecated:: 0.5.0

        - ``data`` is deprecated, it is repalced by:

        >>> dataset = CoauthorPhysicsDataset()
        >>> graph = dataset[0]

    Coauthor CS and Coauthor Physics are co-authorship graphs based on the Microsoft Academic Graph
    from the KDD Cup 2016 challenge. Here, nodes are authors, that are connected by an edge if they
    co-authored a paper; node features represent paper keywords for each author’s papers, and class
    labels indicate most active fields of study for each author.

    Reference: `<https://github.com/shchur/gnn-benchmark#datasets>`_

    Statistics

    - Nodes: 34,493
    - Edges: 495,924 (note that the original dataset has 247,962 edges but DGL adds
      the reverse edges and remove the duplicates, hence with a different number)
    - Number of classes: 5
    - Node feature size: 8,415

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node.
    data : list
        A list of DGLGraph objects

    Examples
    --------
    >>> data = CoauthorPhysicsDataset()
    >>> g = data[0]
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    @property
    def num_classes(self): # -> Literal[5]:
        """Number of classes.

        Return
        -------
        int
        """
        ...
    


class AmazonCoBuyComputerDataset(GNNBenchmarkDataset):
    r""" 'Computer' part of the AmazonCoBuy dataset for node classification task.

    .. deprecated:: 0.5.0

        - ``data`` is deprecated, it is repalced by:

        >>> dataset = AmazonCoBuyComputerDataset()
        >>> graph = dataset[0]

    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley et al., 2015],
    where nodes represent goods, edges indicate that two goods are frequently bought together, node
    features are bag-of-words encoded product reviews, and class labels are given by the product category.

    Reference: `<https://github.com/shchur/gnn-benchmark#datasets>`_

    Statistics:

    - Nodes: 13,752
    - Edges: 491,722 (note that the original dataset has 245,778 edges but DGL adds
      the reverse edges and remove the duplicates, hence with a different number)
    - Number of classes: 10
    - Node feature size: 767

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node.
    data : list
        A list of DGLGraph objects

    Examples
    --------
    >>> data = AmazonCoBuyComputerDataset()
    >>> g = data[0]
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    @property
    def num_classes(self): # -> Literal[10]:
        """Number of classes.

        Return
        -------
        int
        """
        ...
    


class AmazonCoBuyPhotoDataset(GNNBenchmarkDataset):
    r"""AmazonCoBuy dataset for node classification task.

    .. deprecated:: 0.5.0

        - ``data`` is deprecated, it is repalced by:

        >>> dataset = AmazonCoBuyPhotoDataset()
        >>> graph = dataset[0]

    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley et al., 2015],
    where nodes represent goods, edges indicate that two goods are frequently bought together, node
    features are bag-of-words encoded product reviews, and class labels are given by the product category.

    Reference: `<https://github.com/shchur/gnn-benchmark#datasets>`_

    Statistics

    - Nodes: 7,650
    - Edges: 238,163 (note that the original dataset has 119,043 edges but DGL adds
      the reverse edges and remove the duplicates, hence with a different number)
    - Number of classes: 8
    - Node feature size: 745

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node.
    data : list
        A list of DGLGraph objects

    Examples
    --------
    >>> data = AmazonCoBuyPhotoDataset()
    >>> g = data[0]
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    @property
    def num_classes(self): # -> Literal[8]:
        """Number of classes.

        Return
        -------
        int
        """
        ...
    


class CoraFull(CoraFullDataset):
    def __init__(self, **kwargs) -> None:
        ...
    


def AmazonCoBuy(name): # -> AmazonCoBuyComputerDataset | AmazonCoBuyPhotoDataset:
    ...

def Coauthor(name): # -> CoauthorCSDataset | CoauthorPhysicsDataset:
    ...

