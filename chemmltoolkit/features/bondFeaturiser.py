from typing import Any, List, Tuple
from chemmltoolkit.features.featuriser import Featuriser
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol
from rdkit.Chem import Bond


class BondFeaturiser(Featuriser):
    """Generator for bond-based features.

    Args:
        features: A list of features to generate.
    """
    def __init__(self, features: list):
        super(BondFeaturiser, self).__init__(features)

    def process_bond(self, bond: Bond):
        """Generates features for an individual bond.

        Args:
            bond: The bond to featurise.

        Returns:
            A list of features.
        """
        return self._process(bond)

    def process_molecule(self, mol: Mol) -> List[Tuple[int, int, List[Any]]]:
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
            self._process(bond)) for bond in bonds]

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

    def get_feature_lengths(self) -> List[int]:
        """Calculates the length of each feature

        Returns:
            A list of the lengths of each feature.
        """
        molecule = MolFromSmiles('CC')
        bond = molecule.GetBonds()[0]
        return self._get_feature_lengths(bond)
