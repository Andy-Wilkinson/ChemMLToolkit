"""
This type stub file was generated by pyright.
"""

from collections.abc import MutableMapping

"""Views of DGLGraph."""
NodeSpace = ...
EdgeSpace = ...
class HeteroNodeView:
    """A NodeView class to act as G.nodes for a DGLHeteroGraph."""
    __slots__ = ...
    def __init__(self, graph, typeid_getter) -> None:
        ...
    
    def __getitem__(self, key): # -> NodeSpace:
        ...
    
    def __call__(self, ntype=...):
        """Return the nodes."""
        ...
    


class HeteroNodeDataView(MutableMapping):
    """The data view class when G.ndata[ntype] is called."""
    __slots__ = ...
    def __init__(self, graph, ntype, ntid, nodes) -> None:
        ...
    
    def __getitem__(self, key): # -> dict[Unknown, Unknown]:
        ...
    
    def __setitem__(self, key, val): # -> None:
        ...
    
    def __delitem__(self, key): # -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __iter__(self): # -> Iterator[Unknown]:
        ...
    
    def keys(self):
        ...
    
    def values(self):
        ...
    
    def __repr__(self): # -> str:
        ...
    


class HeteroEdgeView:
    """A EdgeView class to act as G.edges for a DGLHeteroGraph."""
    __slots__ = ...
    def __init__(self, graph) -> None:
        ...
    
    def __getitem__(self, key): # -> EdgeSpace:
        ...
    
    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        ...
    


class HeteroEdgeDataView(MutableMapping):
    """The data view class when G.edata[etype] is called."""
    __slots__ = ...
    def __init__(self, graph, etype, edges) -> None:
        ...
    
    def __getitem__(self, key): # -> dict[Unknown, Unknown]:
        ...
    
    def __setitem__(self, key, val): # -> None:
        ...
    
    def __delitem__(self, key): # -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __iter__(self): # -> Iterator[Unknown]:
        ...
    
    def keys(self):
        ...
    
    def values(self):
        ...
    
    def __repr__(self): # -> str:
        ...
    


