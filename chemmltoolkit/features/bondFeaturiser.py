from chemmltoolkit.utils.list_utils import flatten
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol
from rdkit.Chem import Bond


class BondFeaturiser:
    """Generator for bond-based features.

    Args:
        features: A list of features to generate.
    """
    def __init__(self, features):
        self.features = features

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
