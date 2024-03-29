"""
This type stub file was generated by pyright.
"""

from .base import BuiltinFunction

"""Built-in reducer function."""
class ReduceFunction(BuiltinFunction):
    """Base builtin reduce function class."""
    @property
    def name(self):
        """Return the name of this builtin function."""
        ...
    


class SimpleReduceFunction(ReduceFunction):
    """Builtin reduce function that aggregates a single field into another
    single field."""
    def __init__(self, name, msg_field, out_field) -> None:
        ...
    
    @property
    def name(self):
        ...
    


__all__ = []
