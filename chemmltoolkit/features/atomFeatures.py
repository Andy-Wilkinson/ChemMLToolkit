from chemmltoolkit.features.decorators import tokenizable_feature
from rdkit.Chem import Atom
from rdkit.Chem import HybridizationType


def atomic_number(atom: Atom):
    """Atomic number (int).
    """
    return atom.GetAtomicNum()


def charge(atom: Atom):
    """Formal charge (int).
    """
    return atom.GetFormalCharge()


def degree(atom: Atom):
    """Number of directly bonded neighbours (int).
    """
    return atom.GetDegree()


@tokenizable_feature([HybridizationType.SP,
                      HybridizationType.SP2,
                      HybridizationType.SP3,
                      HybridizationType.SP3D,
                      HybridizationType.SP3D2])
def hybridization(atom: Atom):
    """Hybridisation (HybridizationType).
    """
    return atom.GetHybridization()


def hydrogens(atom: Atom):
    """Total number of hydrogen atoms (int).
    """
    return atom.GetTotalNumHs()


def is_aromatic(atom: Atom):
    """If the atom is aromatic (0 or 1).
    """
    return int(atom.GetIsAromatic())


def is_ring(atom: Atom):
    """If the atom is is in a ring (0 or 1).
    """
    return int(atom.IsInRing())


def is_ringsize(ringSize: int):
    """If the atom is is in a ring of the specified size (0 or 1).

    Args:
        ringSize: The size of the ring.
    """
    def _is_ringsize(atom: Atom): return int(atom.IsInRingSize(ringSize))
    return _is_ringsize


def isotope(atom: Atom):
    """Isotope (int).
    """
    return atom.GetIsotope()


def radical(atom: Atom):
    """Number of radical electrons (int).
    """
    return atom.GetNumRadicalElectrons()


def symbol(atom: Atom):
    """Atomic symbol (string).
    """
    return atom.GetSymbol()
