from chemmltoolkit.features.decorators import tokenizable_feature
from rdkit.Chem import Bond
from rdkit.Chem import BondStereo
from rdkit.Chem import BondType


@tokenizable_feature([BondType.SINGLE,
                      BondType.DOUBLE,
                      BondType.TRIPLE,
                      BondType.AROMATIC])
def bond_type(bond: Bond) -> BondType:
    """Type of bond (BondType).
    """
    return bond.GetBondType()


def index(bond: Bond) -> int:
    """Index within the parent molecule (int).
    """
    return bond.GetIdx()


def is_aromatic(bond: Bond) -> int:
    """If the bond is is_aromatic (0 or 1).
    """
    return int(bond.GetIsAromatic())


def is_conjugated(bond: Bond) -> int:
    """If the bond is conjugated (0 or 1).
    """
    return int(bond.GetIsConjugated())


def is_ring(bond: Bond) -> int:
    """If the bond is is in a ring (0 or 1).
    """
    return int(bond.IsInRing())


def is_ringsize(ringSize: int) -> int:
    """If the bond is is in a ring of the specified size (0 or 1).

    Args:
        ringSize: The size of the ring.
    """
    def _is_ringsize(bond: Bond):
        return int(bond.IsInRingSize(ringSize))
    _is_ringsize.__name__ = f'is_ringsize({ringSize})'
    return _is_ringsize


def order(bond: Bond) -> float:
    """Bond order (float).
    """
    return bond.GetBondTypeAsDouble()


@tokenizable_feature([BondStereo.STEREONONE,
                      BondStereo.STEREOANY,
                      BondStereo.STEREOZ,
                      BondStereo.STEREOE])
def stereochemistry(bond: Bond) -> BondStereo:
    """Stereochemistry (BondStereo).
    """
    return bond.GetStereo()
