"""
This type stub file was generated by pyright.
"""

"""Define distributed tensor."""
DIST_TENSOR_ID = ...
class DistTensor:
    ''' Distributed tensor.

    ``DistTensor`` references to a distributed tensor sharded and stored in a cluster of machines.
    It has the same interface as Pytorch Tensor to access its metadata (e.g., shape and data type).
    To access data in a distributed tensor, it supports slicing rows and writing data to rows.
    It does not support any operators of a deep learning framework, such as addition and
    multiplication.

    Currently, distributed tensors are designed to store node data and edge data of a distributed
    graph. Therefore, their first dimensions have to be the number of nodes or edges in the graph.
    The tensors are sharded in the first dimension based on the partition policy of nodes
    or edges. When a distributed tensor is created, the partition policy is automatically
    determined based on the first dimension if the partition policy is not provided. If the first
    dimension matches the number of nodes of a node type, ``DistTensor`` will use the partition
    policy for this particular node type; if the first dimension matches the number of edges of
    an edge type, ``DistTensor`` will use the partition policy for this particular edge type.
    If DGL cannot determine the partition policy automatically (e.g., multiple node types or
    edge types have the same number of nodes or edges), users have to explicity provide
    the partition policy.

    A distributed tensor can be ether named or anonymous.
    When a distributed tensor has a name, the tensor can be persistent if ``persistent=True``.
    Normally, DGL destroys the distributed tensor in the system when the ``DistTensor`` object
    goes away. However, a persistent tensor lives in the system even if
    the ``DistTenor`` object disappears in the trainer process. The persistent tensor has
    the same life span as the DGL servers. DGL does not allow an anonymous tensor to be persistent.

    When a ``DistTensor`` object is created, it may reference to an existing distributed tensor or
    create a new one. A distributed tensor is identified by the name passed to the constructor.
    If the name exists, ``DistTensor`` will reference the existing one.
    In this case, the shape and the data type must match the existing tensor.
    If the name doesn't exist, a new tensor will be created in the kvstore.

    When a distributed tensor is created, its values are initialized to zero. Users
    can define an initialization function to control how the values are initialized.
    The init function has two input arguments: shape and data type and returns a tensor.
    Below shows an example of an init function:

    .. highlight:: python
    .. code-block:: python

        def init_func(shape, dtype):
            return torch.ones(shape=shape, dtype=dtype)

    Parameters
    ----------
    shape : tuple
        The shape of the tensor. The first dimension has to be the number of nodes or
        the number of edges of a distributed graph.
    dtype : dtype
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
    name : string, optional
        The name of the embeddings. The name can uniquely identify embeddings in a system
        so that another ``DistTensor`` object can referent to the distributed tensor.
    init_func : callable, optional
        The function to initialize data in the tensor. If the init function is not provided,
        the values of the embeddings are initialized to zero.
    part_policy : PartitionPolicy, optional
        The partition policy of the rows of the tensor to different machines in the cluster.
        Currently, it only supports node partition policy or edge partition policy.
        The system determines the right partition policy automatically.
    persistent : bool
        Whether the created tensor lives after the ``DistTensor`` object is destroyed.
    is_gdata : bool
        Whether the created tensor is a ndata/edata or not.

    Examples
    --------
    >>> init = lambda shape, dtype: th.ones(shape, dtype=dtype)
    >>> arr = dgl.distributed.DistTensor((g.number_of_nodes(), 2), th.int32, init_func=init)
    >>> print(arr[0:3])
    tensor([[1, 1],
            [1, 1],
            [1, 1]], dtype=torch.int32)
    >>> arr[0:3] = th.ones((3, 2), dtype=th.int32) * 2
    >>> print(arr[0:3])
    tensor([[2, 2],
            [2, 2],
            [2, 2]], dtype=torch.int32)

    Note
    ----
    The creation of ``DistTensor`` is a synchronized operation. When a trainer process tries to
    create a ``DistTensor`` object, the creation succeeds only when all trainer processes
    do the same.
    '''
    def __init__(self, shape, dtype, name=..., init_func=..., part_policy=..., persistent=..., is_gdata=...) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __getitem__(self, idx):
        ...
    
    def __setitem__(self, idx, val): # -> None:
        ...
    
    def __len__(self):
        ...
    
    @property
    def part_policy(self):
        '''Return the partition policy

        Returns
        -------
        PartitionPolicy
            The partition policy of the distributed tensor.
        '''
        ...
    
    @property
    def shape(self):
        '''Return the shape of the distributed tensor.

        Returns
        -------
        tuple
            The shape of the distributed tensor.
        '''
        ...
    
    @property
    def dtype(self):
        '''Return the data type of the distributed tensor.

        Returns
        ------
        dtype
            The data type of the tensor.
        '''
        ...
    
    @property
    def name(self): # -> str:
        '''Return the name of the distributed tensor

        Returns
        -------
        str
            The name of the tensor.
        '''
        ...
    
    @property
    def tensor_name(self):
        '''Return the tensor name

        Returns
        -------
        str
            The name of the tensor.
        '''
        ...
    
    def count_nonzero(self):
        '''Count and return the number of nonzero value

        Returns
        -------
        int
            the number of nonzero value
        '''
        ...
    


