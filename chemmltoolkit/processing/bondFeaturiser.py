from chemmltoolkit.utils.list_utils import flatten
from chemmltoolkit.utils.list_utils import one_hot
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol
from rdkit.Chem import Bond
from rdkit.Chem import BondType


class BondFeaturiser:
    """Generator for a wide range of bond-based features.

    A number of features are implemented,
        - bondtype_onehot: The type of bond (one-hot)
        - is_conjugated: If the bond is conjugated (0 or 1)
        - is_ring: If the bond is in a ring (0 or 1)
        - is_ring3: If the bond is in a 3-membered ring (0 or 1)
        - is_ring4: If the bond is in a 4-membered ring (0 or 1)
        - is_ring5: If the bond is in a 5-membered ring (0 or 1)
        - is_ring6: If the bond is in a 6-membered ring (0 or 1)
        - is_ring7: If the bond is in a 7-membered ring (0 or 1)

    Args:
        features: A list of feature name strings to generate.
        tokens_bondtype: List of tokens to use for the bond type.
            (defaults to single, double, triple, aromatic)
    """
    def __init__(self,
                 features: list,
                 tokens_bondtype: list = [
                     BondType.SINGLE,
                     BondType.DOUBLE,
                     BondType.TRIPLE,
                     BondType.AROMATIC,
                 ]):
        self.features = [self._get_feature(feature) for feature in features]
        self.tokens_bondtype = tokens_bondtype

    def process_bond(self, bond: Bond):
        """Generates features for an individual bond.

        Args:
            bond: The bond to featurise.

        Returns:
            A list of features.
        """

        features = [feature(bond) for feature in self.features]
        return flatten(features)

    def process_molecule(self, mol: Mol):
        """Generates features for all bond in a molecule.

        Args:
            mol: The molecule to featurise.

        Returns:
            A list of tuples for all bonds. Each tuple contains
                (from_atom_index, to_atom_index, features)
        """
        bonds = mol.GetBonds()
        return [(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            self.process_bond(bond)) for bond in bonds]

    def generate_adjacency_matricies(self, mol: Mol):
        """Generates adjacency matricies for a molecule.

        Args:
            mol: The molecule to featurise.

        Returns:
            A list of adjacency matricies, each for an individual feature
        """
        num_atoms = mol.GetNumAtoms()
        bond_features = self.process_molecule(mol)
        num_features = len(bond_features[0][2])

        adj = np.zeros((num_features, num_atoms, num_atoms))

        for atom_begin, atom_end, features in bond_features:
            adj[:, atom_begin, atom_end] = features
            adj[:, atom_end, atom_begin] = features

        return adj

    def get_feature_length(self) -> int:
        """Calculates the length of the generated feature vector

        Returns:
            The length of the feature vector.
        """
        molecule = MolFromSmiles('CC')
        bond = molecule.GetBonds()[0]
        features = self.process_bond(bond)
        return len(features)

    def _get_feature(self, name):
        func_name = f'_f_{name}'
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise f'Undefined bond feature: {name}'

    def _f_bondtype_onehot(self, bond: Bond):
        return one_hot(bond.GetBondType(), self.tokens_bondtype)

    def _f_is_conjugated(self, bond: Bond): return int(bond.GetIsConjugated())
    def _f_is_ring(self, bond: Bond): return int(bond.IsInRing())
    def _f_is_ring3(self, bond: Bond): return int(bond.IsInRingSize(3))
    def _f_is_ring4(self, bond: Bond): return int(bond.IsInRingSize(4))
    def _f_is_ring5(self, bond: Bond): return int(bond.IsInRingSize(5))
    def _f_is_ring6(self, bond: Bond): return int(bond.IsInRingSize(6))
    def _f_is_ring7(self, bond: Bond): return int(bond.IsInRingSize(7))
