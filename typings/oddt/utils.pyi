"""
This type stub file was generated by pyright.
"""

"""Common utilities for ODDT"""
def is_molecule(obj): # -> bool:
    """Check whether an object is an `oddt.toolkits.{rdk,ob}.Molecule` instance.

    .. versionadded:: 0.6
    """
    ...

def is_openbabel_molecule(obj): # -> bool:
    """Check whether an object is an `oddt.toolkits.ob.Molecule` instance.

    .. versionadded:: 0.6
    """
    ...

def is_rdkit_molecule(obj): # -> bool:
    """Check whether an object is an `oddt.toolkits.rdk.Molecule` instance.

    .. versionadded:: 0.6
    """
    ...

def check_molecule(mol, force_protein=..., force_coords=..., non_zero_atoms=...): # -> None:
    """Universal validator of molecule objects. Usage of positional arguments is
    allowed only for molecule object, otherwise it is prohibitted (i.e. the
    order of arguments **will** change). Desired properties of molecule are
    validated based on specified arguments. By default only the object type is
    checked. In case of discrepancy to the specification a `ValueError` is
    raised with appropriate message.

    .. versionadded:: 0.6

    Parameters
    ----------
    mol: oddt.toolkit.Molecule object
        Object to verify

    force_protein: bool (default=False)
        Force the molecule to be marked as protein (mol.protein).

    force_coords: bool (default=False)
        Force the molecule to have non-zero coordinates.

    non_zero_atoms: bool (default=False)
        Check if molecule has at least one atom.

    """
    ...

def compose_iter(iterable, funcs): # -> list[Unknown]:
    """Chain functions and apply them to iterable, by exhausting the iterable.
    Functions are executed in the order from funcs.

    .. versionadded:: 0.6
    """
    ...

def chunker(iterable, chunksize=...): # -> Generator[list[Unknown], None, None]:
    """Generate chunks from a generator object. If iterable is passed which is
    not a generator it will be converted to one with `iter()`.

    .. versionadded:: 0.6
    """
    ...

def method_caller(obj, methodname, *args, **kwargs): # -> Any:
    """Helper function to workaround Python 2 pickle limitations to parallelize
    methods and generator objects"""
    ...

