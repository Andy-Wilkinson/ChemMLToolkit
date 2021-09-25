"""
This type stub file was generated by pyright.
"""

from typing import Generator
from sklearn.utils.deprecation import deprecated

backend = ...
image_backend = ...
image_size = ...
typetable = ...
if __version__ >= '3.0.0':
    GetVdwRad = ...
else:
    GetVdwRad = ...


def readfile(format: str,
             filename: str,
             #  opt=...,
             #  lazy=...
             ) -> Generator[Molecule, None, None]:
    ...


class Molecule(pybel.Molecule):
    def __init__(self, OBMol=..., source=..., protein=...) -> None:
        ...

    @property
    def OBMol(self):
        ...

    @OBMol.setter
    def OBMol(self, value):  # -> None:
        ...

    @property
    def atoms(self):  # -> AtomStack:
        ...

    @property
    def bonds(self):  # -> BondStack:
        ...

    @property
    def coords(self):  # -> ndarray[Unknown, Unknown]:
        ...

    @coords.setter
    def coords(self, new):  # -> None:
        ...

    @property
    def charges(self):  # -> ndarray[Unknown, Unknown]:
        ...

    @property
    def smiles(self):  # -> bytes:
        ...

    # -> bytes:
    def write(self, format=..., filename=..., overwrite=..., opt=..., size=...):
        ...

    @property
    def residues(self):  # -> ResidueStack:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self):  # -> str | bytes | None:
        ...

    @property
    def protein(self):  # -> Unknown:
        """
        A flag for identifing the protein molecules, for which `atom_dict`
        procedures may differ.
        """
        ...

    @protein.setter
    def protein(self, protein):  # -> None:
        """atom_dict caches must be cleared due to property change"""
        ...

    def addh(self, only_polar=...):  # -> None:
        """Add hydrogens"""
        ...

    def removeh(self):  # -> None:
        """Remove hydrogens"""
        ...

    def make3D(self, forcefield=..., steps=...):  # -> None:
        """Generate 3D coordinates"""
        ...

    def make2D(self):  # -> None:
        """Generate 2D coordinates for molecule"""
        ...

    def calccharges(self, model=...):  # -> None:
        """Calculate partial charges for a molecule. By default the Gasteiger
        charge model is used.

        Parameters
        ----------
        model : str (default="gasteiger")
            Method for generating partial charges. Supported models:
            * gasteiger
            * mmff94
            * others supported by OpenBabel (`obabel -L charges`)
        """
        ...

    def __getattr__(self, attr):
        ...

    @property
    def num_rotors(self):  # -> int:
        """Number of strict rotatable """
        ...

    @property
    def canonic_order(self):  # -> NDArray[signedinteger[Any]]:
        """ Returns np.array with canonic order of heavy atoms in the molecule """
        ...

    @property
    def atom_dict(self):  # -> ndarray[Unknown, Unknown] | None:
        ...

    @property
    def res_dict(self):  # -> ndarray[Unknown, Unknown] | None:
        ...

    @property
    def ring_dict(self):  # -> ndarray[Unknown, Unknown] | None:
        ...

    @property
    def clone(self):  # -> Molecule:
        ...

    def clone_coords(self, source):  # -> Molecule:
        ...

    # -> dict[str, Unknown | str | bytes | dict[Unknown, Unknown] | dict[str, ndarray[Unknown, Unknown] | Unknown | None]]:
    def __getstate__(self):
        ...

    def __setstate__(self, state):  # -> None:
        ...


# -> list[Unknown]:
def diverse_conformers_generator(mol, n_conf=..., method=..., seed=..., **kwargs):
    """Produce diverse conformers using current conformer as starting point.
    Returns a generator. Each conformer is a copy of original molecule object.

    .. versionadded:: 0.6

    Parameters
    ----------
    mol : oddt.toolkit.Molecule object
        Molecule for which generating conformers

    n_conf : int (default=10)
        Targer number of conformers

    method : string (default='confab')
        Method for generating conformers. Supported methods:
        * confab
        * ga

    seed : None or int (default=None)
        Random seed

    mutability : int (default=5)
        The inverse of probability of mutation. By default 5, which translates
        to 1/5 (20%) chance of mutation. This setting only works with genetic
        algorithm method ("ga").

    convergence : int (default=5)
        The number of generations with unchanged fitness, should the algorothm
        converge. This setting only works with genetic algorithm method ("ga").

    rmsd : float (default=0.5)
        The conformers are pruned unless their RMSD is higher than this cutoff.
        This setting only works with Confab method ("confab").

    nconf : int (default=10000)
        The number of initial conformers to generate before energy pruning.
        This setting only works with Confab method ("confab").

    energy_gap : float (default=5000.)
        Energy gap from the lowest energy conformer to the highest possible.
        This setting only works with Confab method ("confab").

    Returns
    -------
    mols : list of oddt.toolkit.Molecule objects
        Molecules with diverse conformers
    """
    ...


class AtomStack:
    def __init__(self, OBMol) -> None:
        ...

    def __iter__(self):  # -> Generator[Atom, None, None]:
        ...

    def __len__(self):
        ...

    def __getitem__(self, i):  # -> Atom:
        ...


class Atom(pybel.Atom):
    @property
    @deprecated('RDKit is 0-based and OpenBabel is 1-based. ' 'State which convention you desire and use `idx0` or `idx1`.')
    def idx(self):
        """Note that this index is 1-based as OpenBabel's internal index."""
        ...

    @property
    def idx1(self):
        """Note that this index is 1-based as OpenBabel's internal index."""
        ...

    @property
    def idx0(self):
        """Note that this index is 0-based and OpenBabel's internal index in
        1-based. Changed to be compatible with RDKit"""
        ...

    @property
    def neighbors(self):  # -> list[Atom]:
        ...

    @property
    def residue(self):  # -> Residue:
        ...

    @property
    def bonds(self):  # -> list[Bond]:
        ...


class BondStack:
    def __init__(self, OBMol) -> None:
        ...

    def __iter__(self):  # -> Generator[Bond, None, None]:
        ...

    def __len__(self):
        ...

    def __getitem__(self, i):  # -> Bond:
        ...


class Bond:
    def __init__(self, OBBond) -> None:
        ...

    @property
    def order(self):
        ...

    @property
    def atoms(self):  # -> tuple[Atom, Atom]:
        ...

    @property
    def isrotor(self):
        ...


class Residue:
    """Represent a Pybel residue.

    Required parameter:
       OBResidue -- an Open Babel OBResidue

    Attributes:
       atoms, idx, name.

    (refer to the Open Babel library documentation for more info).

    The original Open Babel atom can be accessed using the attribute:
       OBResidue
    """

    def __init__(self, OBResidue) -> None:
        ...

    @property
    def atoms(self):  # -> list[Atom]:
        """List of Atoms in the Residue"""
        ...

    @property
    @deprecated('Use `idx0` instead.')
    def idx(self):
        """Internal index (0-based) of the Residue"""
        ...

    @property
    def idx0(self):
        """Internal index (0-based) of the Residue"""
        ...

    @property
    def number(self):
        """Residue number"""
        ...

    @property
    def chain(self):
        """Resdiue chain ID"""
        ...

    @property
    def name(self):
        """Residue name"""
        ...

    def __iter__(self):  # -> Iterator[Atom]:
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print(atom)
        """
        ...


class ResidueStack:
    def __init__(self, OBMol) -> None:
        ...

    def __iter__(self):  # -> Generator[Residue, None, None]:
        ...

    def __len__(self):
        ...

    def __getitem__(self, i):  # -> Residue:
        ...


class MoleculeData(pybel.MoleculeData):
    def to_dict(self):  # -> dict[Unknown, Unknown]:
        ...


class Outputfile(pybel.Outputfile):
    def __init__(self, format, filename, overwrite=..., opt=...) -> None:
        ...


class Fingerprint(pybel.Fingerprint):
    @property
    def raw(self):  # -> ndarray[Unknown, Unknown]:
        ...


class Smarts(pybel.Smarts):
    def __init__(self, smartspattern) -> None:
        """Initialise with a SMARTS pattern."""
        ...

    def match(self, molecule):
        """ Checks if there is any match. Returns True or False """
        ...

    def findall(self, molecule, unique=...):  # -> list[Unknown] | list[Any]:
        """Find all matches of the SMARTS pattern to a particular molecule """
        ...
