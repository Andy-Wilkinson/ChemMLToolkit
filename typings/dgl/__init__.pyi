"""
This type stub file was generated by pyright.
"""

import socket
from .backend import backend_name, load_backend
from . import container, contrib, cuda, dataloading, distributed, function, ops, optim, random, sampling
from ._ffi.runtime_ctypes import TypeCode
from ._ffi.function import extract_ext_funcs, get_global_func, list_global_func_names, register_func
from ._ffi.base import DGLError, __version__
from .base import ALL, EID, ETYPE, NID, NTYPE
from .readout import *
from .batch import *
from .convert import *
from .generators import *
from .heterograph import DGLHeteroGraph
from .subgraph import *
from .traversal import *
from .transform import *
from .propagate import *
from .random import *
from .data.utils import load_graphs, save_graphs
from ._deprecate.graph import DGLGraph
# from ._deprecate.graph import DGLGraph as DGLGraphStale
from ._deprecate.nodeflow import *

"""
The ``dgl`` package contains data structure for storing structural and feature data
(i.e., the :class:`DGLGraph` class) and also utilities for generating, manipulating
and transforming graphs.
"""
