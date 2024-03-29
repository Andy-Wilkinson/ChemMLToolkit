"""
This type stub file was generated by pyright.
"""

from .dgl_dataset import DGLBuiltinDataset

""" PPIDataset for inductive learning. """
class PPIDataset(DGLBuiltinDataset):
    r""" Protein-Protein Interaction dataset for inductive node classification

    .. deprecated:: 0.5.0

        - ``lables`` is deprecated, it is replaced by:

            >>> dataset = PPIDataset()
            >>> for g in dataset:
            ....    labels = g.ndata['label']
            ....
            >>>

        - ``features`` is deprecated, it is replaced by:

            >>> dataset = PPIDataset()
            >>> for g in dataset:
            ....    features = g.ndata['feat']
            ....
            >>>

    A toy Protein-Protein Interaction network dataset. The dataset contains
    24 graphs. The average number of nodes per graph is 2372. Each node has
    50 features and 121 labels. 20 graphs for training, 2 for validation
    and 2 for testing.

    Reference: `<http://snap.stanford.edu/graphsage/>`_

    Statistics:

    - Train examples: 20
    - Valid examples: 2
    - Test examples: 2

    Parameters
    ----------
    mode : str
        Must be one of ('train', 'valid', 'test').
        Default: 'train'
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset.
        Default: False
    verbose: bool
        Whether to print out progress information.
        Default: True.

    Attributes
    ----------
    num_labels : int
        Number of labels for each node
    labels : Tensor
        Node labels
    features : Tensor
        Node features

    Examples
    --------
    >>> dataset = PPIDataset(mode='valid')
    >>> num_labels = dataset.num_labels
    >>> for g in dataset:
    ....    feat = g.ndata['feat']
    ....    label = g.ndata['label']
    ....    # your code here
    >>>
    """
    def __init__(self, mode=..., raw_dir=..., force_reload=..., verbose=...) -> None:
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
    def num_labels(self): # -> Literal[121]:
        ...
    
    @property
    def labels(self): # -> Any:
        ...
    
    @property
    def features(self): # -> Any:
        ...
    
    def __len__(self): # -> int:
        """Return number of samples in this dataset."""
        ...
    
    def __getitem__(self, item):
        """Get the item^th sample.

        Parameters
        ---------
        item : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features and node labels.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
        """
        ...
    


class LegacyPPIDataset(PPIDataset):
    """Legacy version of PPI Dataset
    """
    def __getitem__(self, item): # -> tuple[Unknown, Unknown, Unknown]:
        """Get the item^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, Tensor, Tensor)
            The graph, features and its label.
        """
        ...
    


