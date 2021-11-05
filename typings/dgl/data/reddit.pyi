"""
This type stub file was generated by pyright.
"""

from .dgl_dataset import DGLBuiltinDataset

""" Reddit dataset for community detection """
class RedditDataset(DGLBuiltinDataset):
    r""" Reddit dataset for community detection (node classification)

    .. deprecated:: 0.5.0

        - ``graph`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]

        - ``num_labels`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> num_classes = dataset.num_classes

        - ``train_mask`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.ndata['train_mask']

        - ``val_mask`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.ndata['val_mask']

        - ``test_mask`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.ndata['test_mask']

        - ``features`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> features = graph.ndata['feat']

        - ``labels`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> labels = graph.ndata['label']

    This is a graph dataset from Reddit posts made in the month of September, 2014.
    The node label in this case is the community, or “subreddit”, that a post belongs to.
    The authors sampled 50 large communities and built a post-to-post graph, connecting
    posts if the same user comments on both. In total this dataset contains 232,965
    posts with an average degree of 492. We use the first 20 days for training and the
    remaining days for testing (with 30% used for validation).

    Reference: `<http://snap.stanford.edu/graphsage/>`_

    Statistics

    - Nodes: 232,965
    - Edges: 114,615,892
    - Node feature size: 602
    - Number of training samples: 153,431
    - Number of validation samples: 23,831
    - Number of test samples: 55,703

    Parameters
    ----------
    self_loop : bool
        Whether load dataset with self loop connections. Default: False
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
        Number of classes for each node
    graph : :class:`dgl.DGLGraph`
        Graph of the dataset
    num_labels : int
        Number of classes for each node
    train_mask: numpy.ndarray
        Mask of training nodes
    val_mask: numpy.ndarray
        Mask of validation nodes
    test_mask: numpy.ndarray
        Mask of test nodes
    features : Tensor
        Node features
    labels :  Tensor
        Node labels

    Examples
    --------
    >>> data = RedditDataset()
    >>> g = data[0]
    >>> num_classes = data.num_classes
    >>>
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>>
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> test_mask = g.ndata['test_mask']
    >>>
    >>> # get labels
    >>> label = g.ndata['label']
    >>>
    >>> # Train, Validation and Test
    """
    def __init__(self, self_loop=..., raw_dir=..., force_reload=..., verbose=...) -> None:
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
    def num_classes(self): # -> Literal[41]:
        r"""Number of classes for each node."""
        ...
    
    @property
    def num_labels(self): # -> Literal[41]:
        ...
    
    @property
    def graph(self): # -> DGLHeteroGraph:
        ...
    
    @property
    def train_mask(self):
        ...
    
    @property
    def val_mask(self):
        ...
    
    @property
    def test_mask(self):
        ...
    
    @property
    def features(self): # -> dict[Unknown, Unknown]:
        ...
    
    @property
    def labels(self): # -> dict[Unknown, Unknown]:
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
            graph structure, node labels, node features and splitting masks:

            - ``ndata['label']``: node label
            - ``ndata['feat']``: node feature
            - ``ndata['train_mask']``： mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']:`` mask for test node set
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""Number of graphs in the dataset"""
        ...
    


