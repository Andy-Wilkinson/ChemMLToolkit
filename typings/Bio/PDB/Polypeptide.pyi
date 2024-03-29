from typing import List

"""
This type stub file was generated by pyright.
"""

"""Polypeptide-related classes (construction and representation).

Simple example with multiple chains,

    >>> from Bio.PDB.PDBParser import PDBParser
    >>> from Bio.PDB.Polypeptide import PPBuilder
    >>> structure = PDBParser().get_structure('2BEG', 'PDB/2BEG.pdb')
    >>> ppb=PPBuilder()
    >>> for pp in ppb.build_peptides(structure):
    ...     print(pp.get_sequence())
    LVFFAEDVGSNKGAIIGLMVGGVVIA
    LVFFAEDVGSNKGAIIGLMVGGVVIA
    LVFFAEDVGSNKGAIIGLMVGGVVIA
    LVFFAEDVGSNKGAIIGLMVGGVVIA
    LVFFAEDVGSNKGAIIGLMVGGVVIA

Example with non-standard amino acids using HETATM lines in the PDB file,
in this case selenomethionine (MSE):

    >>> from Bio.PDB.PDBParser import PDBParser
    >>> from Bio.PDB.Polypeptide import PPBuilder
    >>> structure = PDBParser().get_structure('1A8O', 'PDB/1A8O.pdb')
    >>> ppb=PPBuilder()
    >>> for pp in ppb.build_peptides(structure):
    ...     print(pp.get_sequence())
    DIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNW
    TETLLVQNANPDCKTILKALGPGATLEE
    TACQG

If you want to, you can include non-standard amino acids in the peptides:

    >>> for pp in ppb.build_peptides(structure, aa_only=False):
    ...     print(pp.get_sequence())
    ...     print("%s %s" % (pp.get_sequence()[0], pp[0].get_resname()))
    ...     print("%s %s" % (pp.get_sequence()[-7], pp[-7].get_resname()))
    ...     print("%s %s" % (pp.get_sequence()[-6], pp[-6].get_resname()))
    MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQG
    M MSE
    M MSE
    M MSE

In this case the selenomethionines (the first and also seventh and sixth from
last residues) have been shown as M (methionine) by the get_sequence method.
"""
standard_aa_names: List[str] = ...
aa1 = ...
aa3 = ...
d1_to_index = ...
dindex_to_1 = ...
d3_to_index = ...
dindex_to_3 = ...


def index_to_one(index):
    """Index to corresponding one letter amino acid name.

    >>> index_to_one(0)
    'A'
    >>> index_to_one(19)
    'Y'
    """
    ...


def one_to_index(s):
    """One letter code to index.

    >>> one_to_index('A')
    0
    >>> one_to_index('Y')
    19
    """
    ...


def index_to_three(i):
    """Index to corresponding three letter amino acid name.

    >>> index_to_three(0)
    'ALA'
    >>> index_to_three(19)
    'TYR'
    """
    ...


def three_to_index(s):
    """Three letter code to index.

    >>> three_to_index('ALA')
    0
    >>> three_to_index('TYR')
    19
    """
    ...


def three_to_one(s: str) -> str:
    """Three letter code to one letter code.

    >>> three_to_one('ALA')
    'A'
    >>> three_to_one('TYR')
    'Y'

    For non-standard amino acids, you get a KeyError:

    >>> three_to_one('MSE')
    Traceback (most recent call last):
       ...
    KeyError: 'MSE'
    """
    ...


def one_to_three(s):
    """One letter code to three letter code.

    >>> one_to_three('A')
    'ALA'
    >>> one_to_three('Y')
    'TYR'
    """
    ...


def is_aa(residue, standard=...):  # -> bool:
    """Return True if residue object/string is an amino acid.

    :param residue: a L{Residue} object OR a three letter amino acid code
    :type residue: L{Residue} or string

    :param standard: flag to check for the 20 AA (default false)
    :type standard: boolean

    >>> is_aa('ALA')
    True

    Known three letter codes for modified amino acids are supported,

    >>> is_aa('FME')
    True
    >>> is_aa('FME', standard=True)
    False
    """
    ...


class Polypeptide(list):
    """A polypeptide is simply a list of L{Residue} objects."""

    def get_ca_list(self):  # -> list[Unknown]:
        """Get list of C-alpha atoms in the polypeptide.

        :return: the list of C-alpha atoms
        :rtype: [L{Atom}, L{Atom}, ...]
        """
        ...

    def get_phi_psi_list(self):  # -> list[Unknown]:
        """Return the list of phi/psi dihedral angles."""
        ...

    def get_tau_list(self):  # -> list[Unknown]:
        """List of tau torsions angles for all 4 consecutive Calpha atoms."""
        ...

    def get_theta_list(self):  # -> list[Unknown]:
        """List of theta angles for all 3 consecutive Calpha atoms."""
        ...

    def get_sequence(self):  # -> Seq:
        """Return the AA sequence as a Seq object.

        :return: polypeptide sequence
        :rtype: L{Seq}
        """
        ...

    def __repr__(self):  # -> str:
        """Return string representation of the polypeptide.

        Return <Polypeptide start=START end=END>, where START
        and END are sequence identifiers of the outer residues.
        """
        ...


class _PPBuilder:
    """Base class to extract polypeptides.

    It checks if two consecutive residues in a chain are connected.
    The connectivity test is implemented by a subclass.

    This assumes you want both standard and non-standard amino acids.
    """

    def __init__(self, radius) -> None:
        """Initialize the base class.

        :param radius: distance
        :type radius: float
        """
        ...

    def build_peptides(self, entity, aa_only=...):  # -> list[Unknown]:
        """Build and return a list of Polypeptide objects.

        :param entity: polypeptides are searched for in this object
        :type entity: L{Structure}, L{Model} or L{Chain}

        :param aa_only: if 1, the residue needs to be a standard AA
        :type aa_only: int
        """
        ...


class CaPPBuilder(_PPBuilder):
    """Use CA--CA distance to find polypeptides."""

    def __init__(self, radius=...) -> None:
        """Initialize the class."""
        ...


class PPBuilder(_PPBuilder):
    """Use C--N distance to find polypeptides."""

    def __init__(self, radius=...) -> None:
        """Initialize the class."""
        ...
