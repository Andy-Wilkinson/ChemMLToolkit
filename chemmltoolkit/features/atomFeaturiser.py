from chemmltoolkit.utils.list_utils import flatten
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol
from rdkit.Chem import Atom


class AtomFeaturiser:
    """Generator for atom-based features.
    Args:
        features: A list of features to generate.
    """
    def __init__(self, features: list):
        self.features = features

    def process_atom(self, atom: Atom):
        """Generates features for an individual atom.

        Args:
            atom: The atom to featurise.

        Returns:
            A list of features.
        """

        features = [feature(atom) for feature in self.features]
        return flatten(features)

    def process_molecule(self, mol: Mol):
        """Generates features for all atoms in a molecule.

        Args:
            mol: The molecule to featurise.

        Returns:
            A nested list of features for all atoms.
        """
        atoms = mol.GetAtoms()
        return [self.process_atom(atom) for atom in atoms]

    def get_feature_length(self) -> int:
        """Calculates the length of the generated feature vector

        Returns:
            The length of the feature vector.
        """
        molecule = MolFromSmiles('CC')
        atom = molecule.GetAtoms()[0]
        features = self.process_atom(atom)
        return len(features)
