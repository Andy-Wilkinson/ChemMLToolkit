"""
This type stub file was generated by pyright.
"""

import abc
from .dgl_dataset import DGLBuiltinDataset

"""RDF datasets
Datasets from "A Collection of Benchmark Datasets for
Systematic Evaluations of Machine Learning on
the Semantic Web"
"""
__all__ = ['AIFB', 'MUTAG', 'BGS', 'AM', 'AIFBDataset', 'MUTAGDataset', 'BGSDataset', 'AMDataset']
RENAME_DICT = ...
class Entity:
    """Class for entities
    Parameters
    ----------
    id : str
        ID of this entity
    cls : str
        Type of this entity
    """
    def __init__(self, e_id, cls) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class Relation:
    """Class for relations
    Parameters
    ----------
    cls : str
        Type of this relation
    """
    def __init__(self, cls) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class RDFGraphDataset(DGLBuiltinDataset):
    """Base graph dataset class from RDF tuples.

    To derive from this, implement the following abstract methods:
    * ``parse_entity``
    * ``parse_relation``
    * ``process_tuple``
    * ``process_idx_file_line``
    * ``predict_category``
    Preprocessed graph and other data will be cached in the download folder
    to speedup data loading.
    The dataset should contain a "trainingSet.tsv" and a "testSet.tsv" file
    for training and testing samples.

    Attributes
    ----------
    graph : dgl.DGLraph
        Graph structure
    num_classes : int
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction
    train_idx : Tensor
        Entity IDs for training. All IDs are local IDs w.r.t. to ``predict_category``.
    test_idx : Tensor
        Entity IDs for testing. All IDs are local IDs w.r.t. to ``predict_category``.
    labels : Tensor
        All the labels of the entities in ``predict_category``

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str or path
        URL to download the raw dataset.
    predict_category : str
        Predict category.
    print_every : int, optional
        Preprocessing log for every X tuples.
    insert_reverse : bool, optional
        If true, add reverse edge and reverse relations to the final graph.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Ignore cached pre-processed data.
    verbose: bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, name, url, predict_category, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def process(self): # -> None:
        ...
    
    def load_raw_tuples(self, root_path): # -> chain[Unknown]:
        """Loading raw RDF dataset

        Parameters
        ----------
        root_path : str
            Root path containing the data

        Returns
        -------
            Loaded rdf data
        """
        ...
    
    def process_raw_tuples(self, raw_tuples, root_path):
        """Processing raw RDF dataset

        Parameters
        ----------
        raw_tuples:
            Raw rdf tuples
        root_path: str
            Root path containing the data
        """
        ...
    
    def build_graph(self, mg, src, dst, ntid, etid, ntypes, etypes):
        """Build the graphs

        Parameters
        ----------
        mg: MultiDiGraph
            Input graph
        src: Numpy array
            Source nodes
        dst: Numpy array
            Destination nodes
        ntid: Numpy array
            Node types for each node
        etid: Numpy array
            Edge types for each edge
        ntypes: list
            Node types
        etypes: list
            Edge types

        Returns
        -------
        g: DGLGraph
        """
        ...
    
    def load_data_split(self, ent2id, root_path): # -> tuple[ndarray[Unknown, Unknown], ndarray[Unknown, Unknown], ndarray[Unknown, Unknown], int]:
        """Load data split

        Parameters
        ----------
        ent2id: func
            A function mapping entity to id
        root_path: str
            Root path containing the data

        Return
        ------
        train_idx: Numpy array
            Training set
        test_idx: Numpy array
            Testing set
        labels: Numpy array
            Labels
        num_classes: int
            Number of classes
        """
        ...
    
    def parse_idx_file(self, filename, ent2id, label_dict, labels): # -> list[Unknown]:
        """Parse idx files

        Parameters
        ----------
        filename: str
            File to parse
        ent2id: func
            A function mapping entity to id
        label_dict: dict
            Map label to label id
        labels: dict
            Map entity id to label id

        Return
        ------
        idx: list
            Entity idss
        """
        ...
    
    def has_cache(self): # -> bool:
        """check if there is a processed data"""
        ...
    
    def save(self): # -> None:
        """save the graph list and the labels"""
        ...
    
    def load(self): # -> None:
        """load the graph list and the labels from disk"""
        ...
    
    def __getitem__(self, idx):
        r"""Gets the graph object
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset."""
        ...
    
    @property
    def save_name(self):
        ...
    
    @property
    def graph(self):
        ...
    
    @property
    def predict_category(self): # -> Any:
        ...
    
    @property
    def num_classes(self): # -> int | Any:
        ...
    
    @property
    def train_idx(self):
        ...
    
    @property
    def test_idx(self):
        ...
    
    @property
    def labels(self):
        ...
    
    @abc.abstractmethod
    def parse_entity(self, term): # -> None:
        """Parse one entity from an RDF term.
        Return None if the term does not represent a valid entity and the
        whole tuple should be ignored.
        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term
        Returns
        -------
        Entity or None
            An entity.
        """
        ...
    
    @abc.abstractmethod
    def parse_relation(self, term): # -> None:
        """Parse one relation from an RDF term.
        Return None if the term does not represent a valid relation and the
        whole tuple should be ignored.
        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term
        Returns
        -------
        Relation or None
            A relation
        """
        ...
    
    @abc.abstractmethod
    def process_tuple(self, raw_tuple, sbj, rel, obj): # -> None:
        """Process the tuple.
        Return (Entity, Relation, Entity) tuple for as the final tuple.
        Return None if the tuple should be ignored.

        Parameters
        ----------
        raw_tuple : tuple of rdflib.term.Identifier
            (subject, predicate, object) tuple
        sbj : Entity
            Subject entity
        rel : Relation
            Relation
        obj : Entity
            Object entity
        Returns
        -------
        (Entity, Relation, Entity)
            The final tuple or None if should be ignored
        """
        ...
    
    @abc.abstractmethod
    def process_idx_file_line(self, line): # -> None:
        """Process one line of ``trainingSet.tsv`` or ``testSet.tsv``.
        Parameters
        ----------
        line : str
            One line of the file
        Returns
        -------
        (str, str)
            One sample and its label
        """
        ...
    


class AIFBDataset(RDFGraphDataset):
    r"""AIFB dataset for node classification task

    .. deprecated:: 0.5.0

        - ``graph`` is deprecated, it is replaced by:

            >>> dataset = AIFBDataset()
            >>> graph = dataset[0]

        - ``train_idx`` is deprecated, it can be replaced by:

            >>> dataset = AIFBDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.nodes[dataset.category].data['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()

        - ``test_idx`` is deprecated, it can be replaced by:

            >>> dataset = AIFBDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.nodes[dataset.category].data['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    AIFB DataSet is a Semantic Web (RDF) dataset used as a benchmark in
    data mining.  It records the organizational structure of AIFB at the
    University of Karlsruhe.

    AIFB dataset statistics:

    - Nodes: 7262
    - Edges: 48810 (including reverse edges)
    - Target Category: Personen
    - Number of Classes: 4
    - Label Split:

        - Train: 140
        - Test: 36

    Parameters
    -----------
    print_every: int
        Preprocessing log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
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
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction
    labels : Tensor
        All the labels of the entities in ``predict_category``
    graph : :class:`dgl.DGLGraph`
        Graph structure
    train_idx : Tensor
        Entity IDs for training. All IDs are local IDs w.r.t. to ``predict_category``.
    test_idx : Tensor
        Entity IDs for testing. All IDs are local IDs w.r.t. to ``predict_category``.

    Examples
    --------
    >>> dataset = dgl.data.rdf.AIFBDataset()
    >>> graph = dataset[0]
    >>> category = dataset.predict_category
    >>> num_classes = dataset.num_classes
    >>>
    >>> train_mask = g.nodes[category].data.pop('train_mask')
    >>> test_mask = g.nodes[category].data.pop('test_mask')
    >>> labels = g.nodes[category].data.pop('labels')
    """
    entity_prefix = ...
    relation_prefix = ...
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, AIFBDataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['test_mask']``: mask for testing node set
            - ``ndata['labels']``: mask for labels
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset.

        Return
        -------
        int
        """
        ...
    
    def parse_entity(self, term): # -> Entity | None:
        ...
    
    def parse_relation(self, term): # -> Relation | None:
        ...
    
    def process_tuple(self, raw_tuple, sbj, rel, obj): # -> tuple[Unknown, Unknown, Unknown] | None:
        ...
    
    def process_idx_file_line(self, line): # -> tuple[Unknown, Unknown]:
        ...
    


class AIFB(AIFBDataset):
    """AIFB dataset. Same as AIFBDataset.
    """
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    


class MUTAGDataset(RDFGraphDataset):
    r"""MUTAG dataset for node classification task

    .. deprecated:: 0.5.0

        - ``graph`` is deprecated, it is replaced by:

            >>> dataset = MUTAGDataset()
            >>> graph = dataset[0]

        - ``train_idx`` is deprecated, it can be replaced by:

            >>> dataset = MUTAGDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.nodes[dataset.category].data['train_mask']
            >>> train_idx = th.nonzero(train_mask).squeeze()

        - ``test_idx`` is deprecated, it can be replaced by:

            >>> dataset = MUTAGDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.nodes[dataset.category].data['test_mask']
            >>> test_idx = th.nonzero(test_mask).squeeze()

    Mutag dataset statistics:

    - Nodes: 27163
    - Edges: 148100 (including reverse edges)
    - Target Category: d
    - Number of Classes: 2
    - Label Split:

        - Train: 272
        - Test: 68

    Parameters
    -----------
    print_every: int
        Preprocessing log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
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
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction
    labels : Tensor
        All the labels of the entities in ``predict_category``
    graph : :class:`dgl.DGLGraph`
        Graph structure
    train_idx : Tensor
        Entity IDs for training. All IDs are local IDs w.r.t. to ``predict_category``.
    test_idx : Tensor
        Entity IDs for testing. All IDs are local IDs w.r.t. to ``predict_category``.

    Examples
    --------
    >>> dataset = dgl.data.rdf.MUTAGDataset()
    >>> graph = dataset[0]
    >>> category = dataset.predict_category
    >>> num_classes = dataset.num_classes
    >>>
    >>> train_mask = g.nodes[category].data.pop('train_mask')
    >>> test_mask = g.nodes[category].data.pop('test_mask')
    >>> labels = g.nodes[category].data.pop('labels')
    """
    d_entity = ...
    bond_entity = ...
    entity_prefix = ...
    relation_prefix = ...
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, MUTAGDataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['test_mask']``: mask for testing node set
            - ``ndata['labels']``: mask for labels
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset.

        Return
        -------
        int
        """
        ...
    
    def parse_entity(self, term): # -> Entity | None:
        ...
    
    def parse_relation(self, term): # -> Relation | None:
        ...
    
    def process_tuple(self, raw_tuple, sbj, rel, obj): # -> tuple[Unknown, Unknown, Unknown] | None:
        ...
    
    def process_idx_file_line(self, line): # -> tuple[Unknown, Unknown]:
        ...
    


class MUTAG(MUTAGDataset):
    """MUTAG dataset. Same as MUTAGDataset.
    """
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    


class BGSDataset(RDFGraphDataset):
    r"""BGS dataset for node classification task

    .. deprecated:: 0.5.0

        - ``graph`` is deprecated, it is replaced by:

            >>> dataset = BGSDataset()
            >>> graph = dataset[0]

        - ``train_idx`` is deprecated, it can be replaced by:

            >>> dataset = BGSDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.nodes[dataset.category].data['train_mask']
            >>> train_idx = th.nonzero(train_mask).squeeze()

        - ``test_idx`` is deprecated, it can be replaced by:

            >>> dataset = BGSDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.nodes[dataset.category].data['test_mask']
            >>> test_idx = th.nonzero(test_mask).squeeze()

    BGS namespace convention:
    ``http://data.bgs.ac.uk/(ref|id)/<Major Concept>/<Sub Concept>/INSTANCE``.
    We ignored all literal nodes and the relations connecting them in the
    output graph. We also ignored the relation used to mark whether a
    term is CURRENT or DEPRECATED.

    BGS dataset statistics:

    - Nodes: 94806
    - Edges: 672884 (including reverse edges)
    - Target Category: Lexicon/NamedRockUnit
    - Number of Classes: 2
    - Label Split:

        - Train: 117
        - Test: 29

    Parameters
    -----------
    print_every: int
        Preprocessing log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
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
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction
    labels : Tensor
        All the labels of the entities in ``predict_category``
    graph : :class:`dgl.DGLGraph`
        Graph structure
    train_idx : Tensor
        Entity IDs for training. All IDs are local IDs w.r.t. to ``predict_category``.
    test_idx : Tensor
        Entity IDs for testing. All IDs are local IDs w.r.t. to ``predict_category``.

    Examples
    --------
    >>> dataset = dgl.data.rdf.BGSDataset()
    >>> graph = dataset[0]
    >>> category = dataset.predict_category
    >>> num_classes = dataset.num_classes
    >>>
    >>> train_mask = g.nodes[category].data.pop('train_mask')
    >>> test_mask = g.nodes[category].data.pop('test_mask')
    >>> labels = g.nodes[category].data.pop('labels')
    """
    entity_prefix = ...
    status_prefix = ...
    relation_prefix = ...
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, BGSDataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['test_mask']``: mask for testing node set
            - ``ndata['labels']``: mask for labels
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset.

        Return
        -------
        int
        """
        ...
    
    def parse_entity(self, term): # -> Entity | None:
        ...
    
    def parse_relation(self, term): # -> Relation | None:
        ...
    
    def process_tuple(self, raw_tuple, sbj, rel, obj): # -> tuple[Unknown, Unknown, Unknown] | None:
        ...
    
    def process_idx_file_line(self, line): # -> tuple[Unknown, Unknown]:
        ...
    


class BGS(BGSDataset):
    """BGS dataset. Same as BGSDataset.
    """
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    


class AMDataset(RDFGraphDataset):
    """AM dataset. for node classification task

    .. deprecated:: 0.5.0

        - ``graph`` is deprecated, it is replaced by:

            >>> dataset = AMDataset()
            >>> graph = dataset[0]

        - ``train_idx`` is deprecated, it can be replaced by:

            >>> dataset = AMDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.nodes[dataset.category].data['train_mask']
            >>> train_idx = th.nonzero(train_mask).squeeze()

        - ``test_idx`` is deprecated, it can be replaced by:

            >>> dataset = AMDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.nodes[dataset.category].data['test_mask']
            >>> test_idx = th.nonzero(test_mask).squeeze()

    Namespace convention:

    - Instance: ``http://purl.org/collections/nl/am/<type>-<id>``
    - Relation: ``http://purl.org/collections/nl/am/<name>``

    We ignored all literal nodes and the relations connecting them in the
    output graph.

    AM dataset statistics:

    - Nodes: 881680
    - Edges: 5668682 (including reverse edges)
    - Target Category: proxy
    - Number of Classes: 11
    - Label Split:

        - Train: 802
        - Test: 198

    Parameters
    -----------
    print_every: int
        Preprocessing log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
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
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction
    labels : Tensor
        All the labels of the entities in ``predict_category``
    graph : :class:`dgl.DGLGraph`
        Graph structure
    train_idx : Tensor
        Entity IDs for training. All IDs are local IDs w.r.t. to ``predict_category``.
    test_idx : Tensor
        Entity IDs for testing. All IDs are local IDs w.r.t. to ``predict_category``.

    Examples
    --------
    >>> dataset = dgl.data.rdf.AMDataset()
    >>> graph = dataset[0]
    >>> category = dataset.predict_category
    >>> num_classes = dataset.num_classes
    >>>
    >>> train_mask = g.nodes[category].data.pop('train_mask')
    >>> test_mask = g.nodes[category].data.pop('test_mask')
    >>> labels = g.nodes[category].data.pop('labels')
    """
    entity_prefix = ...
    relation_prefix = ...
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    
    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, AMDataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['test_mask']``: mask for testing node set
            - ``ndata['labels']``: mask for labels
        """
        ...
    
    def __len__(self): # -> Literal[1]:
        r"""The number of graphs in the dataset.

        Return
        -------
        int
        """
        ...
    
    def parse_entity(self, term): # -> Entity | None:
        ...
    
    def parse_relation(self, term): # -> Relation | None:
        ...
    
    def process_tuple(self, raw_tuple, sbj, rel, obj): # -> tuple[Unknown, Unknown, Unknown] | None:
        ...
    
    def process_idx_file_line(self, line): # -> tuple[Unknown, Unknown]:
        ...
    


class AM(AMDataset):
    """AM dataset. Same as AMDataset.
    """
    def __init__(self, print_every=..., insert_reverse=..., raw_dir=..., force_reload=..., verbose=...) -> None:
        ...
    


if __name__ == '__main__':
    dataset = ...