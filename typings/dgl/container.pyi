"""
This type stub file was generated by pyright.
"""

from ._ffi.object import ObjectBase, register_object

"""Container data structures used in DGL runtime.
reference: tvm/python/tvm/collections.py
"""
@register_object
class List(ObjectBase):
    """List container of DGL.

    You do not need to create List explicitly.
    Normally python list and tuple will be converted automatically
    to List during dgl function call.
    You may get List in return values of DGL function call.
    """
    def __getitem__(self, i):
        ...
    
    def __len__(self):
        ...
    


@register_object
class Map(ObjectBase):
    """Map container of DGL.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during dgl function call.
    You can use convert to create a dict[ObjectBase-> ObjectBase] into a Map
    """
    def __getitem__(self, k):
        ...
    
    def __contains__(self, k):
        ...
    
    def items(self): # -> list[tuple[Unknown, Unknown]]:
        """Get the items from the map"""
        ...
    
    def __len__(self):
        ...
    


@register_object
class StrMap(Map):
    """A special map container that has str as key.

    You can use convert to create a dict[str->ObjectBase] into a Map.
    """
    def items(self): # -> list[tuple[Unknown, Unknown]]:
        """Get the items from the map"""
        ...
    


@register_object
class Value(ObjectBase):
    """Object wrapper for various values."""
    @property
    def data(self):
        """Return the value data."""
        ...
    


def convert_to_strmap(value): # -> dict[Unknown, Unknown]:
    """Convert a python dictionary to a dgl.contrainer.StrMap"""
    ...

