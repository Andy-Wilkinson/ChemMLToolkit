"""
This type stub file was generated by pyright.
"""

from sklearn.utils.deprecation import deprecated
from rdkit import Chem

"""
rdkit - A Cinfony module for accessing the RDKit from CPython

Global variables:
  Chem and AllChem - the underlying RDKit Python bindings
  informats - a dictionary of supported input formats
  outformats - a dictionary of supported output formats
  descs - a list of supported descriptors
  fps - a list of supported fingerprint types
  forcefields - a list of supported forcefields
"""
_descDict = ...
backend = ...
__version__ = ...
image_backend = ...
image_size = ...
elementtable = ...
SMARTS_DEF = ...
fps = ...
descs = ...
_formats = ...
_notinformats = ...
_notoutformats = ...
if not Chem.INCHI_AVAILABLE:
    ...
informats = ...
outformats = ...
base_feature_factory = ...
_forcefields = ...
forcefields = ...
def readfile(format, filename, lazy=..., opt=..., **kwargs): # -> Generator[Molecule, None, None]:
    """Iterate over the molecules in a file.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       filename

    You can access the first molecule in a file using the next() method
    of the iterator:
        mol = next(readfile("smi", "myfile.smi"))

    You can make a list of the molecules in a file using:
        mols = list(readfile("smi", "myfile.smi"))

    You can iterate over the molecules in a file as shown in the
    following code snippet:
    >>> atomtotal = 0
    >>> for mol in readfile("sdf", "head.sdf"):
    ...     atomtotal += len(mol.atoms)
    ...
    >>> print(atomtotal)
    43
    """
    ...

def readstring(format, string, **kwargs): # -> Molecule:
    """Read in a molecule from a string.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       string

    Example:
    >>> input = "C1=CC=CS1"
    >>> mymol = readstring("smi", input)
    >>> len(mymol.atoms)
    5
    """
    ...

class Outputfile:
    """Represent a file to which *output* is to be sent.

    Required parameters:
       format - see the outformats variable for a list of available
                output formats
       filename

    Optional parameters:
       overwite -- if the output file already exists, should it
                   be overwritten? (default is False)

    Methods:
       write(molecule)
       close()
    """
    def __init__(self, format, filename, overwrite=...) -> None:
        ...
    
    def write(self, molecule): # -> None:
        """Write a molecule to the output file.

        Required parameters:
           molecule
        """
        ...
    
    def close(self): # -> None:
        """Close the Outputfile to further writing."""
        ...
    


class Molecule:
    """Represent an rdkit Molecule.

    Required parameter:
       Mol -- an RDKit Mol or any type of cinfony Molecule

    Attributes:
       atoms, data, formula, molwt, title

    Methods:
       addh(), calcfp(), calcdesc(), draw(), localopt(), make3D(), removeh(),
       write()

    The underlying RDKit Mol can be accessed using the attribute:
       Mol
    """
    _cinfony = ...
    def __new__(cls, Mol=..., source=..., *args, **kwargs): # -> Any | None:
        """ Trap RDKit molecules which are 'None' """
        ...
    
    def __init__(self, Mol=..., source=..., protein=...) -> None:
        ...
    
    @property
    def Mol(self): # -> None:
        ...
    
    @Mol.setter
    def Mol(self, value): # -> None:
        ...
    
    @property
    def atoms(self): # -> AtomStack:
        ...
    
    @property
    def data(self): # -> MoleculeData:
        ...
    
    @property
    def molwt(self):
        ...
    
    @property
    def formula(self):
        ...
    
    title = ...
    @property
    def coords(self): # -> ndarray[Unknown, Unknown]:
        ...
    
    @coords.setter
    def coords(self, new): # -> None:
        ...
    
    @property
    def charges(self): # -> ndarray[Unknown, Unknown]:
        ...
    
    @property
    def smiles(self):
        ...
    
    @property
    def residues(self): # -> ResidueStack:
        ...
    
    @property
    def protein(self): # -> Unknown:
        """
        A flag for identifing the protein molecules, for which `atom_dict`
        procedures may differ.
        """
        ...
    
    @protein.setter
    def protein(self, protein): # -> None:
        """atom_dict caches must be cleared due to property change"""
        ...
    
    @property
    def sssr(self): # -> list[list[Unknown]]:
        ...
    
    @property
    def num_rotors(self):
        ...
    
    @property
    def bonds(self): # -> BondStack:
        ...
    
    @property
    def canonic_order(self): # -> ndarray[Unknown, Unknown]:
        """ Returns np.array with canonic order of heavy atoms in the molecule """
        ...
    
    @property
    def atom_dict(self): # -> ndarray[Unknown, Unknown] | None:
        ...
    
    @property
    def res_dict(self): # -> ndarray[Unknown, Unknown] | None:
        ...
    
    @property
    def ring_dict(self): # -> ndarray[Unknown, Unknown] | None:
        ...
    
    @property
    def clone(self): # -> Molecule:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> str | bytes | None:
        ...
    
    def clone_coords(self, source): # -> Molecule:
        ...
    
    def addh(self, only_polar=..., **kwargs): # -> None:
        """Add hydrogens."""
        ...
    
    def removeh(self, **kwargs): # -> None:
        """Remove hydrogens."""
        ...
    
    def write(self, format=..., filename=..., overwrite=..., size=..., **kwargs): # -> bytes | str | None:
        """Write the molecule to a file or return a string.

        Optional parameters:
           format -- see the informats variable for a list of available
                     output formats (default is "smi")
           filename -- default is None
           overwite -- if the output file already exists, should it
                       be overwritten? (default is False)

        If a filename is specified, the result is written to a file.
        Otherwise, a string is returned containing the result.

        To write multiple molecules to the same file you should use
        the Outputfile class.
        """
        ...
    
    def __iter__(self): # -> Iterator[Atom]:
        """Iterate over the Atoms of the Molecule.

        This allows constructions such as the following:
           for atom in mymol:
               print(atom)
        """
        ...
    
    def calcdesc(self, descnames=...): # -> dict[Unknown, Unknown]:
        """Calculate descriptor values.

        Optional parameter:
           descnames -- a list of names of descriptors

        If descnames is not specified, all available descriptors are
        calculated. See the descs variable for a list of available
        descriptors.
        """
        ...
    
    def calcfp(self, fptype=..., opt=...): # -> Fingerprint:
        """Calculate a molecular fingerprint.

        Optional parameters:
           fptype -- the fingerprint type (default is "rdkit"). See the
                     fps variable for a list of of available fingerprint
                     types.
           opt -- a dictionary of options for fingerprints. Currently only used
                  for radius and bitInfo in Morgan fingerprints.
        """
        ...
    
    def calccharges(self, model=...): # -> None:
        """Calculate partial charges for a molecule. By default the Gasteiger
        charge model is used.

        Parameters
        ----------
        model : str (default="gasteiger")
            Method for generating partial charges. Supported models:
            * gasteiger
            * mmff94
        """
        ...
    
    def localopt(self, forcefield=..., steps=...): # -> None:
        """Locally optimize the coordinates.

        Optional parameters:
           forcefield -- default is "uff". See the forcefields variable
                         for a list of available forcefields.
           steps -- default is 500

        If the molecule does not have any coordinates, make3D() is
        called before the optimization.
        """
        ...
    
    def make3D(self, forcefield=..., steps=...): # -> None:
        """Generate 3D coordinates.

        Optional parameters:
           forcefield -- default is "uff". See the forcefields variable
                         for a list of available forcefields.
           steps -- default is 50

        Once coordinates are generated, a quick
        local optimization is carried out with 50 steps and the
        UFF forcefield. Call localopt() if you want
        to improve the coordinates further.
        """
        ...
    
    def make2D(self): # -> None:
        """Generate 2D coordinates for molecule"""
        ...
    
    def __getstate__(self): # -> dict[str, Unknown | dict[Unknown, Unknown] | dict[str, ndarray[Unknown, Unknown] | Unknown | None] | None] | dict[str, Unknown | dict[str, None] | None]:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    


def diverse_conformers_generator(mol, n_conf=..., method=..., seed=..., rmsd=...): # -> list[Unknown]:
    """Produce diverse conformers using current conformer as starting point.
    Each conformer is a copy of original molecule object.

    .. versionadded:: 0.6

    Parameters
    ----------
    mol : oddt.toolkit.Molecule object
        Molecule for which generating conformers

    n_conf : int (default=10)
        Targer number of conformers

    method : string (default='etkdg')
        Method for generating conformers. Supported methods: "etkdg", "etdg",
        "kdg", "dg".

    seed : None or int (default=None)
        Random seed

    rmsd : float (default=0.5)
        The minimum RMSD that separates conformers to be ratained (otherwise,
        they will be pruned).

    Returns
    -------
    mols : list of oddt.toolkit.Molecule objects
        Molecules with diverse conformers
    """
    ...

class AtomStack:
    def __init__(self, Mol) -> None:
        ...
    
    def __iter__(self): # -> Generator[Atom, None, None]:
        ...
    
    def __len__(self):
        ...
    
    def __getitem__(self, i): # -> Atom:
        ...
    


class Atom:
    """Represent an rdkit Atom.

    Required parameters:
       Atom -- an RDKit Atom

    Attributes:
        atomicnum, coords, formalcharge

    The original RDKit Atom can be accessed using the attribute:
       Atom
    """
    def __init__(self, Atom) -> None:
        ...
    
    @property
    def atomicnum(self):
        ...
    
    @property
    def coords(self): # -> tuple[Literal[0], Literal[0], Literal[0]] | tuple[Unknown, Unknown, Unknown]:
        ...
    
    @property
    def formalcharge(self):
        ...
    
    @property
    @deprecated('RDKit is 0-based and OpenBabel is 1-based. ' 'State which convention you desire and use `idx0` or `idx1`.')
    def idx(self):
        """Note that this index is 1-based and RDKit's internal index in 0-based.
        Changed to be compatible with OpenBabel"""
        ...
    
    @property
    def idx1(self):
        """Note that this index is 1-based and RDKit's internal index in 0-based.
        Changed to be compatible with OpenBabel"""
        ...
    
    @property
    def idx0(self):
        """ Note that this index is 0-based as RDKit's"""
        ...
    
    @property
    def neighbors(self): # -> list[Atom]:
        ...
    
    @property
    def bonds(self): # -> list[Bond]:
        ...
    
    @property
    def partialcharge(self): # -> float:
        ...
    
    def __str__(self) -> str:
        ...
    


class BondStack:
    def __init__(self, Mol) -> None:
        ...
    
    def __iter__(self): # -> Generator[Bond, None, None]:
        ...
    
    def __len__(self):
        ...
    
    def __getitem__(self, i): # -> Bond:
        ...
    


class Bond:
    def __init__(self, Bond) -> None:
        ...
    
    @property
    def order(self):
        ...
    
    @property
    def atoms(self): # -> tuple[Atom, Atom]:
        ...
    
    @property
    def isrotor(self): # -> bool:
        ...
    


class Residue:
    """Represent a RDKit residue.

    Required parameter:
       ParentMol -- Parent molecule (Mol) object
       path -- atoms path of a residue

    Attributes:
       atoms, idx, name.

    (refer to the Open Babel library documentation for more info).

    The Mol object constucted of residues' atoms can be accessed using the attribute:
       Residue
    """
    def __init__(self, ParentMol, atom_path, idx=...) -> None:
        ...
    
    @property
    def atoms(self): # -> list[Atom] | AtomStack:
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
    def number(self): # -> Literal[0]:
        """Residue number"""
        ...
    
    @property
    def chain(self): # -> Literal['']:
        """Resdiue chain ID"""
        ...
    
    @property
    def name(self): # -> Literal['UNL']:
        """Residue name"""
        ...
    
    def __iter__(self): # -> Iterator[Atom]:
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print(atom)
        """
        ...
    


class ResidueStack:
    def __init__(self, Mol, paths) -> None:
        ...
    
    def __iter__(self): # -> Generator[Residue, None, None]:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i): # -> Residue:
        ...
    


class Smarts:
    """A Smarts Pattern Matcher

    Required parameters:
       smartspattern

    Methods:
       findall(molecule)

    Example:
    >>> mol = readstring("smi","CCN(CC)CC") # triethylamine
    >>> smarts = Smarts("[#6][#6]") # Matches an ethyl group
    >>> print(smarts.findall(mol))
    [(0, 1), (3, 4), (5, 6)]

    The numbers returned are the indices (starting from 0) of the atoms
    that match the SMARTS pattern. In this case, there are three matches
    for each of the three ethyl groups in the molecule.
    """
    def __init__(self, smartspattern) -> None:
        """Initialise with a SMARTS pattern."""
        ...
    
    def match(self, molecule):
        """Find all matches of the SMARTS pattern to a particular molecule.

        Required parameters:
           molecule
        """
        ...
    
    def findall(self, molecule, unique=...):
        """Find all matches of the SMARTS pattern to a particular molecule.

        Required parameters:
           molecule
        """
        ...
    


class MoleculeData:
    """Store molecule data in a dictionary-type object

    Required parameters:
      Mol -- an RDKit Mol

    Methods and accessor methods are like those of a dictionary except
    that the data is retrieved on-the-fly from the underlying Mol.

    Example:
    >>> mol = next(readfile("sdf", 'head.sdf'))
    >>> data = mol.data
    >>> print(data)
    {'Comment': 'CORINA 2.61 0041  25.10.2001', 'NSC': '1'}
    >>> print(len(data), data.keys(), data.has_key("NSC"))
    2 ['Comment', 'NSC'] True
    >>> print(data['Comment'])
    CORINA 2.61 0041  25.10.2001
    >>> data['Comment'] = 'This is a new comment'
    >>> for k,v in data.items():
    ...    print(k, "-->", v)
    Comment --> This is a new comment
    NSC --> 1
    >>> del data['NSC']
    >>> print(len(data), data.keys(), data.has_key("NSC"))
    1 ['Comment'] False
    """
    def __init__(self, Mol) -> None:
        ...
    
    def keys(self):
        ...
    
    def values(self): # -> list[Unknown]:
        ...
    
    def items(self): # -> zip[Tuple[Unknown, Unknown]]:
        ...
    
    def __iter__(self): # -> Iterator[Unknown]:
        ...
    
    def iteritems(self): # -> Iterator[Tuple[Unknown, Unknown]]:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __contains__(self, key):
        ...
    
    def __delitem__(self, key): # -> None:
        ...
    
    def clear(self): # -> None:
        ...
    
    def has_key(self, key): # -> bool:
        ...
    
    def update(self, dictionary): # -> None:
        ...
    
    def __getitem__(self, key):
        ...
    
    def __setitem__(self, key, value): # -> None:
        ...
    
    def to_dict(self):
        ...
    
    def __repr__(self):
        ...
    


class Fingerprint:
    """A Molecular Fingerprint.

    Required parameters:
       fingerprint -- a vector calculated by one of the fingerprint methods

    Attributes:
       fp -- the underlying fingerprint object
       bits -- a list of bits set in the Fingerprint

    Methods:
       The "|" operator can be used to calculate the Tanimoto coeff. For example,
       given two Fingerprints 'a', and 'b', the Tanimoto coefficient is given by:
          tanimoto = a | b
    """
    def __init__(self, fingerprint) -> None:
        ...
    
    def __or__(self, other):
        ...
    
    def __getattr__(self, attr): # -> list[Unknown]:
        ...
    
    def __str__(self) -> str:
        ...
    
    @property
    def raw(self): # -> ndarray[Unknown, Unknown]:
        ...
    


if __name__ == "__main__":
    ...
