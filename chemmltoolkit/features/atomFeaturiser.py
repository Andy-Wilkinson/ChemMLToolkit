from typing import Any, List
from chemmltoolkit.features.featuriser import Featuriser
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol
from rdkit.Chem import Atom


class AtomFeaturiser(Featuriser):
    """Generator for atom-based features.
    Args:
        features: A list of features to generate.
    """

    def __init__(self, features: list):
        super(AtomFeaturiser, self).__init__(features)

    def process_atom(self, atom: Atom):
        """Generates features for an individual atom.

        Args:
            atom: The atom to featurise.

        Returns:
            A list of features.
        """
        return self._process(atom)

    def process_molecule(self, mol: Mol) -> List[List[Any]]:
        """Generates features for all atoms in a molecule.

        Args:
            mol: The molecule to featurise.

        Returns:
            A nested list of features for all atoms.
        """
        atoms = mol.GetAtoms()
        return [self._process(atom) for atom in atoms]

    def get_feature_lengths(self) -> List[int]:
        """Calculates the length of each feature

        Returns:
            A list of the lengths of each feature.
        """
        molecule = MolFromSmiles('CC')
        atom = molecule.GetAtoms()[0]
        return self._get_feature_lengths(atom)
