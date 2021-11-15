"""
This type stub file was generated by pyright.
"""

"""Common implementation of Object generic related logic"""
_CLASS_OBJECT_BASE = ...
class ObjectGeneric:
    """Base class for all classes that can be converted to object."""
    def asobject(self):
        """Convert value to object"""
        ...
    


def convert_to_object(value):
    """Convert a python value to corresponding object type.

    Parameters
    ----------
    value : str
        The value to be inspected.

    Returns
    -------
    object : Object
        The corresponding object value.
    """
    ...
