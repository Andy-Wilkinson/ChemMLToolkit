"""
This type stub file was generated by pyright.
"""

from typing import Generator
from Bio.PDB.Chain import Chain
from Bio.PDB.Entity import Entity

"""The structure class, representing a macromolecular structure."""


class Structure(Entity):
    """The Structure class contains a collection of Model instances."""

    def __init__(self, id) -> None:
        """Initialize the class."""
        ...

    def __repr__(self):
        """Return the structure identifier."""
        ...

    def get_models(self):  # -> Generator[Unknown, None, None]:
        """Return models."""
        ...

    def get_chains(self) -> Generator[Chain, None, None]:
        """Return chains from models."""
        ...

    def get_residues(self):  # -> Generator[Unknown, None, None]:
        """Return residues from chains."""
        ...

    def get_atoms(self):  # -> Generator[Unknown, None, None]:
        """Return atoms from residue."""
        ...

    def atom_to_internal_coordinates(self, verbose: bool = ...) -> None:
        """Create/update internal coordinates from Atom X,Y,Z coordinates.

        Internal coordinates are bond length, angle and dihedral angles.

        :param verbose bool: default False
            describe runtime problems

        """
        ...

    def internal_to_atom_coordinates(self, verbose: bool = ...) -> None:
        """Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems

        :raises Exception: if any chain does not have .pic attribute
        """
        ...
