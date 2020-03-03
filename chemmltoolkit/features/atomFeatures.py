from chemmltoolkit.features.decorators import tokenizable_feature
from rdkit.Chem import Atom
from rdkit.Chem import AllChem
from rdkit.Chem import ChiralType
from rdkit.Chem import HybridizationType


def atomic_number(atom: Atom) -> int:
    """Atomic number (int).
    """
    return atom.GetAtomicNum()


def atomic_mass(atom: Atom) -> float:
    """Atomic mass (float).
    """
    return atom.GetMass()


def charge(atom: Atom) -> int:
    """Formal charge (int).
    """
    return atom.GetFormalCharge()


def charge_gasteiger(atom: Atom) -> float:
    """Gasteiger partial charge (float).
    """
    if not atom.HasProp('_GasteigerCharge'):
        mol = atom.GetOwningMol()
        AllChem.ComputeGasteigerCharges(mol)
    return atom.GetDoubleProp('_GasteigerCharge')


def charge_gasteiger_h(atom: Atom) -> float:
    """Gasteiger partial charge for implicit hydrogens (float).
    """
    if not atom.HasProp('_GasteigerHCharge'):
        mol = atom.GetOwningMol()
        AllChem.ComputeGasteigerCharges(mol)
    return atom.GetDoubleProp('_GasteigerHCharge')


@tokenizable_feature([ChiralType.CHI_UNSPECIFIED,
                      ChiralType.CHI_TETRAHEDRAL_CW,
                      ChiralType.CHI_TETRAHEDRAL_CCW,
                      ChiralType.CHI_OTHER])
def chiral_tag(atom: Atom) -> ChiralType:
    """Chirality of the atom (ChiralType)
    """
    return atom.GetChiralTag()


def degree(atom: Atom) -> int:
    """Number of directly bonded neighbours (int).
    """
    return atom.GetDegree()


@tokenizable_feature([HybridizationType.SP,
                      HybridizationType.SP2,
                      HybridizationType.SP3,
                      HybridizationType.SP3D,
                      HybridizationType.SP3D2])
def hybridization(atom: Atom) -> HybridizationType:
    """Hybridisation (HybridizationType).
    """
    return atom.GetHybridization()


def hydrogens(atom: Atom) -> int:
    """Total number of hydrogen atoms (int).
    """
    return atom.GetTotalNumHs()


def index(atom: Atom) -> int:
    """Index within the parent molecule (int).
    """
    return atom.GetIdx()


def is_aromatic(atom: Atom) -> int:
    """If the atom is aromatic (0 or 1).
    """
    return int(atom.GetIsAromatic())


def is_ring(atom: Atom) -> int:
    """If the atom is is in a ring (0 or 1).
    """
    return int(atom.IsInRing())


def is_ringsize(ringSize: int) -> int:
    """If the atom is is in a ring of the specified size (0 or 1).

    Args:
        ringSize: The size of the ring.
    """
    def _is_ringsize(atom: Atom):
        return int(atom.IsInRingSize(ringSize))
    _is_ringsize.__name__ = f'is_ringsize({ringSize})'
    return _is_ringsize


def isotope(atom: Atom) -> int:
    """Isotope (int).
    """
    return atom.GetIsotope()


def radical(atom: Atom) -> int:
    """Number of radical electrons (int).
    """
    return atom.GetNumRadicalElectrons()


def symbol(atom: Atom) -> str:
    """Atomic symbol (string).
    """
    return atom.GetSymbol()
