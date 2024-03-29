"""
This type stub file was generated by pyright.
"""

from __future__ import annotations
import sys
from typing import Any, Dict, Optional
from System.Windows.Forms import Form

"""
pybel - A Cinfony module for accessing Open Babel

Global variables:
  ob - the underlying SWIG bindings for Open Babel
  informats - a dictionary of supported input formats
  outformats - a dictionary of supported output formats
  descs - a list of supported descriptors
  fps - a list of supported fingerprint types
  forcefields - a list of supported forcefields
"""
if sys.platform[: 4] == "java":
    _obfuncs = ...
    _obconsts = ...
else:
    _obdotnet = ...
    _obfuncs = ...
    _obconsts = ...
    _obfuncs = ...
_obconv = ...
_builder = ...
informats = ...
outformats = ...
descs = ...
_descdict = ...
fps = ...
_fingerprinters = ...
forcefields = ...
_forcefields = ...
charges = ...
_charges = ...
operations = ...
_operations = ...
ipython_3d = ...


def readfile(format, filename, opt=...):  # -> Generator[Molecule, None, None]:
    """Iterate over the molecules in a file.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       filename

    Optional parameters:
       opt    - a dictionary of format-specific options
                For format options with no parameters, specify the
                value as None.

    You can access the first molecule in a file using the next() method
    of the iterator (or the next() keyword in Python 3):
        mol = readfile("smi", "myfile.smi").next() # Python 2
        mol = next(readfile("smi", "myfile.smi"))  # Python 3

    You can make a list of the molecules in a file using:
        mols = list(readfile("smi", "myfile.smi"))

    You can iterate over the molecules in a file as shown in the
    following code snippet:
    >>> atomtotal = 0
    >>> for mol in readfile("sdf", "head.sdf"):
    ...     atomtotal += len(mol.atoms)
    ...
    >>> print atomtotal
    43
    """
    ...


def readstring(format: str,
               string: str,
               opt: Dict[str, Any] = ...
               ) -> Molecule:
    """Read in a molecule from a string.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       string

    Optional parameters:
       opt    - a dictionary of format-specific options
                For format options with no parameters, specify the
                value as None.

    Example:
    >>> input = "C1=CC=CS1"
    >>> mymol = readstring("smi", input)
    >>> len(mymol.atoms)
    5
    """
    ...


class Outputfile:
    """Represent a file to which *output* is to be sent.

    Although it's possible to write a single molecule to a file by
    calling the write() method of a molecule, if multiple molecules
    are to be written to the same file you should use the Outputfile
    class.

    Required parameters:
       format - see the outformats variable for a list of available
                output formats
       filename

    Optional parameters:
       overwrite -- if the output file already exists, should it
                   be overwritten? (default is False)
       opt -- a dictionary of format-specific options
              For format options with no parameters, specify the
              value as None.

    Methods:
       write(molecule)
       close()
    """

    def __init__(self, format, filename, overwrite=..., opt=...) -> None:
        ...

    def write(self, molecule):  # -> None:
        """Write a molecule to the output file.

        Required parameters:
           molecule
        """
        ...

    def close(self):  # -> None:
        """Close the Outputfile to further writing."""
        ...

    def __enter__(self):  # -> Outputfile:
        """Called by with statement, returns itself"""
        ...

    def __exit__(self, exc_type, exc_value, traceback):  # -> None:
        """Called by with statement, closes itself"""
        ...


class Molecule:
    """Represent a Pybel Molecule.

    Required parameter:
       OBMol -- an Open Babel OBMol or any type of cinfony Molecule

    Attributes:
       atoms, charge, conformers, data, dim, energy, exactmass, formula,
       molwt, spin, sssr, title, unitcell.
    (refer to the Open Babel library documentation for more info).

    Methods:
       addh(), calcfp(), calcdesc(), draw(), localopt(), make2D(), make3D()
       calccharges(), removeh(), write()

    The underlying Open Babel molecule can be accessed using the attribute:
       OBMol
    """
    _cinfony = ...

    def __init__(self, OBMol) -> None:
        ...

    @property
    def atoms(self):  # -> list[Atom]:
        ...

    @property
    def residues(self):  # -> list[Residue]:
        ...

    @property
    def charge(self):
        ...

    @property
    def conformers(self):
        ...

    @property
    def data(self):  # -> MoleculeData:
        ...

    @property
    def dim(self):
        ...

    @property
    def energy(self):
        ...

    @property
    def exactmass(self):
        ...

    @property
    def formula(self):
        ...

    @property
    def molwt(self):
        ...

    @property
    def spin(self):
        ...

    @property
    def sssr(self):
        ...

    title = ...

    @property
    def unitcell(self):
        ...

    @property
    def clone(self):  # -> Molecule:
        ...

    def __iter__(self):  # -> Iterator[Atom]:
        """Iterate over the Atoms of the Molecule.

        This allows constructions such as the following:
           for atom in mymol:
               print atom
        """
        ...

    def calcdesc(self, descnames=...):  # -> dict[Unknown, Unknown]:
        """Calculate descriptor values.

        Optional parameter:
           descnames -- a list of names of descriptors

        If descnames is not specified, all available descriptors are
        calculated. See the descs variable for a list of available
        descriptors.
        """
        ...

    def calcfp(self, fptype=...):  # -> Fingerprint:
        """Calculate a molecular fingerprint.

        Optional parameters:
           fptype -- the fingerprint type (default is "FP2"). See the
                     fps variable for a list of of available fingerprint
                     types.
        """
        ...

    def calccharges(self, model=...):  # -> list[Unknown]:
        """Estimates atomic partial charges in the molecule.

        Optional parameters:
           model -- default is "mmff94". See the charges variable for a list
                    of available charge models (in shell, `obabel -L charges`)

        This method populates the `partialcharge` attribute of each atom
        in the molecule in place.
        """
        ...

    def write(self,
              format: str = ...,
              filename: Optional[str] = ...,
              overwrite: bool = ...,
              opt: Dict[str, Any] = ...) -> str:
        """Write the molecule to a file or return a string.

        Optional parameters:
           format -- see the informats variable for a list of available
                     output formats (default is "smi")
           filename -- default is None
           overwite -- if the output file already exists, should it
                       be overwritten? (default is False)
           opt -- a dictionary of format specific options
                  For format options with no parameters, specify the
                  value as None.

        If a filename is specified, the result is written to a file.
        Otherwise, a string is returned containing the result.

        To write multiple molecules to the same file you should use
        the Outputfile class.
        """
        ...

    def localopt(self, forcefield=..., steps=...):  # -> None:
        """Locally optimize the coordinates.

        Optional parameters:
           forcefield -- default is "mmff94". See the forcefields variable
                         for a list of available forcefields.
           steps -- default is 500

        If the molecule does not have any coordinates, make3D() is
        called before the optimization. Note that the molecule needs
        to have explicit hydrogens. If not, call addh().
        """
        ...

    def make2D(self):  # -> None:
        """Generate 2D coordinates."""
        ...

    def make3D(self, forcefield=..., steps=...):  # -> None:
        """Generate 3D coordinates.

        Optional parameters:
           forcefield -- default is "mmff94". See the forcefields variable
                         for a list of available forcefields.
           steps -- default is 50

        Once coordinates are generated, hydrogens are added and a quick
        local optimization is carried out with 50 steps and the
        MMFF94 forcefield. Call localopt() if you want
        to improve the coordinates further.
        """
        ...

    def addh(self):  # -> None:
        """Add hydrogens."""
        ...

    def removeh(self):  # -> None:
        """Remove hydrogens."""
        ...

    def convertdbonds(self):  # -> None:
        """Convert Dative Bonds."""
        ...

    def __str__(self) -> str:
        ...

    def draw(self, show=..., filename=..., update=..., usecoords=...):  # -> None:
        """Create a 2D depiction of the molecule.

        Optional parameters:
          show -- display on screen (default is True)
          filename -- write to file (default is None)
          update -- update the coordinates of the atoms to those
                    determined by the structure diagram generator
                    (default is False)
          usecoords -- don't calculate 2D coordinates, just use
                       the current coordinates (default is False)

        Tkinter and Python Imaging Library are required for image display.
        """
        ...


class Atom:
    """Represent a Pybel atom.

    Required parameter:
       OBAtom -- an Open Babel OBAtom

    Attributes:
       atomicmass, atomicnum, cidx, coords, coordidx, degree, exactmass,
       formalcharge, heavydegree, heterodegree, hyb, idx,
       implicitvalence, isotope, partialcharge, residue, spin, type,
       vector.

    (refer to the Open Babel library documentation for more info).

    The original Open Babel atom can be accessed using the attribute:
       OBAtom
    """

    def __init__(self, OBAtom) -> None:
        ...

    @property
    def coords(self):  # -> tuple[Unknown, Unknown, Unknown]:
        ...

    @property
    def atomicmass(self):
        ...

    @property
    def atomicnum(self):
        ...

    @property
    def cidx(self):
        ...

    @property
    def coordidx(self):
        ...

    @property
    def degree(self):
        ...

    @property
    def exactmass(self):
        ...

    @property
    def formalcharge(self):
        ...

    @property
    def heavydegree(self):
        ...

    @property
    def heavyvalence(self):  # -> NoReturn:
        ...

    @property
    def heterodegree(self):
        ...

    @property
    def heterovalence(self):  # -> NoReturn:
        ...

    @property
    def hyb(self):
        ...

    @property
    def idx(self):
        ...

    @property
    def implicitvalence(self):
        ...

    @property
    def isotope(self):
        ...

    @property
    def partialcharge(self):
        ...

    @property
    def residue(self):  # -> Residue:
        ...

    @property
    def spin(self):
        ...

    @property
    def type(self):
        ...

    @property
    def valence(self):  # -> NoReturn:
        ...

    @property
    def vector(self):
        ...

    def __str__(self) -> str:
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
        ...

    @property
    def idx(self):
        ...

    @property
    def name(self):
        ...

    def __iter__(self):  # -> Iterator[Atom]:
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print atom
        """
        ...


class Fingerprint:
    """A Molecular Fingerprint.

    Required parameters:
       fingerprint -- a vector calculated by OBFingerprint.FindFingerprint()

    Attributes:
       fp -- the underlying fingerprint object
       bits -- a list of bits set in the Fingerprint

    Methods:
       The "|" operator can be used to calculate the Tanimoto coeff. For
       example, given two Fingerprints 'a', and 'b', the Tanimoto coefficient
       is given by:
          tanimoto = a | b
    """

    def __init__(self, fingerprint) -> None:
        ...

    def __or__(self, other):
        ...

    @property
    def bits(self):  # -> list[Unknown]:
        ...

    def __str__(self) -> str:
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
    >>> print smarts.findall(mol)
    [(1, 2), (4, 5), (6, 7)]

    The numbers returned are the indices (starting from 1) of the atoms
    that match the SMARTS pattern. In this case, there are three matches
    for each of the three ethyl groups in the molecule.
    """

    def __init__(self, smartspattern) -> None:
        """Initialise with a SMARTS pattern."""
        ...

    def findall(self, molecule):  # -> list[Unknown]:
        """Find all matches of the SMARTS pattern to a particular molecule.

        Required parameters:
           molecule
        """
        ...


class MoleculeData:
    """Store molecule data in a dictionary-type object

    Required parameters:
      obmol -- an Open Babel OBMol

    Methods and accessor methods are like those of a dictionary except
    that the data is retrieved on-the-fly from the underlying OBMol.

    Example:
    >>> mol = readfile("sdf", 'head.sdf').next() # Python 2
    >>> # mol = next(readfile("sdf", 'head.sdf')) # Python 3
    >>> data = mol.data
    >>> print data
    {'Comment': 'CORINA 2.61 0041  25.10.2001', 'NSC': '1'}
    >>> print len(data), data.keys(), data.has_key("NSC")
    2 ['Comment', 'NSC'] True
    >>> print data['Comment']
    CORINA 2.61 0041  25.10.2001
    >>> data['Comment'] = 'This is a new comment'
    >>> for k,v in data.items():
    ...    print k, "-->", v
    Comment --> This is a new comment
    NSC --> 1
    >>> del data['NSC']
    >>> print len(data), data.keys(), data.has_key("NSC")
    1 ['Comment'] False
    """

    def __init__(self, obmol) -> None:
        ...

    def keys(self):  # -> list[Unknown]:
        ...

    def values(self):  # -> list[Unknown]:
        ...

    def items(self):  # -> Iterator[Tuple[Unknown, Unknown]]:
        ...

    def __iter__(self):  # -> Iterator[Unknown]:
        ...

    def iteritems(self):  # -> Iterator[Tuple[Unknown, Unknown]]:
        ...

    def __len__(self):  # -> int:
        ...

    def __contains__(self, key):
        ...

    def __delitem__(self, key):  # -> None:
        ...

    def clear(self):  # -> None:
        ...

    def has_key(self, key):  # -> bool:
        ...

    def update(self, dictionary):  # -> None:
        ...

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value):  # -> None:
        ...

    def __repr__(self):  # -> str:
        ...


if sys.platform[: 3] == "cli":
    class _MyForm(Form):
        def __init__(self) -> None:
            ...

        def setup(self, filename, title):  # -> None:
            ...


if __name__ == "__main__":
    ...
