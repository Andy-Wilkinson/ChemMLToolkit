"""
This type stub file was generated by pyright.
"""

from .._ffi.object import ObjectBase, register_object

"""For HeteroGraph Serialization"""
def tensor_dict_to_ndarray_dict(tensor_dict): # -> dict[Unknown, Unknown]:
    """Convert dict[str, tensor] to StrMap[NDArray]"""
    ...

def save_heterographs(filename, g_list, labels): # -> None:
    """Save heterographs into file"""
    ...

@register_object("heterograph_serialize.HeteroGraphData")
class HeteroGraphData(ObjectBase):
    """Object to hold the data to be stored for DGLHeteroGraph"""
    @staticmethod
    def create(g):
        ...
    
    def get_graph(self): # -> DGLHeteroGraph:
        ...
    

