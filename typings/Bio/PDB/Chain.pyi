"""
This type stub file was generated by pyright.
"""

from typing import Generator, Optional
from Bio.PDB.Entity import Entity
from Bio.PDB.Residue import Residue

"""Chain class, used in Structure objects."""


class Chain(Entity):
    """Define Chain class.

    Chain is an object of type Entity, stores residues and includes a method to
    access atoms from residues.
    """

    def __init__(self, id) -> None:
        """Initialize the class."""
        ...

    def __gt__(self, other) -> bool:
        """Validate if id is greater than other.id."""
        ...

    def __ge__(self, other) -> bool:
        """Validate if id is greater or equal than other.id."""
        ...

    def __lt__(self, other) -> bool:
        """Validate if id is less than other.id."""
        ...

    def __le__(self, other) -> bool:
        """Validate if id is less or equal than other id."""
        ...

    def __getitem__(self, id):
        """Return the residue with given id.

        The id of a residue is (hetero flag, sequence identifier, insertion code).
        If id is an int, it is translated to (" ", id, " ") by the _translate_id
        method.

        Arguments:
         - id - (string, int, string) or int

        """
        ...

    def __contains__(self, id):  # -> bool:
        """Check if a residue with given id is present in this chain.

        Arguments:
         - id - (string, int, string) or int

        """
        ...

    def __delitem__(self, id):  # -> None:
        """Delete item.

        Arguments:
         - id - (string, int, string) or int

        """
        ...

    def __repr__(self):
        """Return the chain identifier."""
        ...

    def get_unpacked_list(self):  # -> list[Unknown]:
        """Return a list of undisordered residues.

        Some Residue objects hide several disordered residues
        (DisorderedResidue objects). This method unpacks them,
        ie. it returns a list of simple Residue objects.
        """
        ...

    def has_id(self, id):  # -> bool:
        """Return 1 if a residue with given id is present.

        The id of a residue is (hetero flag, sequence identifier, insertion code).

        If id is an int, it is translated to (" ", id, " ") by the _translate_id
        method.

        Arguments:
         - id - (string, int, string) or int

        """
        ...

    def get_residues(self) -> Generator[Residue, None, None]:
        """Return residues."""
        ...

    def get_atoms(self):  # -> Generator[Unknown, None, None]:
        """Return atoms from residues."""
        ...

    def atom_to_internal_coordinates(self, verbose: bool = ...) -> None:
        """Create/update internal coordinates from Atom X,Y,Z coordinates.

        Internal coordinates are bond length, angle and dihedral angles.

        :param verbose bool: default False
            describe runtime problems
        """
        ...

    # -> None:
    def internal_to_atom_coordinates(self, verbose: bool = ..., start: Optional[int] = ..., fin: Optional[int] = ...):
        """Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems
        :param: start, fin lists
            sequence position, insert code for begin, end of subregion to
            process
        :raises Exception: if any chain does not have .pic attribute
        """
        ...
