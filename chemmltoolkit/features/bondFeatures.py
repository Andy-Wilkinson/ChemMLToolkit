from chemmltoolkit.features.decorators import tokenizable_feature
from rdkit.Chem import Bond
from rdkit.Chem import BondType


@tokenizable_feature([BondType.SINGLE,
                      BondType.DOUBLE,
                      BondType.TRIPLE,
                      BondType.AROMATIC])
def bond_type(bond: Bond):
    """Type of bond (BondType).
    """
    return bond.GetBondType()


def is_conjugated(bond: Bond):
    """If the bond is conjugated (0 or 1).
    """
    return int(bond.GetIsConjugated())


def is_ring(bond: Bond):
    """If the bond is is in a ring (0 or 1).
    """
    return int(bond.IsInRing())


def is_ringsize(ringSize: int):
    """If the bond is is in a ring of the specified size (0 or 1).

    Args:
        ringSize: The size of the ring.
    """
    def _is_ringsize(bond: Bond): return int(bond.IsInRingSize(ringSize))
    return _is_ringsize
