from typing import List
from chemmltoolkit.features.featuriser import Featuriser
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol


class MoleculeFeaturiser(Featuriser):
    """Generator for molecule-based features.

    Args:
        features: A list of features to generate.
    """
    def __init__(self, features: list):
        super(MoleculeFeaturiser, self).__init__(features)

    def process_molecule(self, mol: Mol):
        """Generates molecular features.

        Args:
            mol: The molecule to featurise.

        Returns:
            A list of features.
        """
        return self._process(mol)

    def get_feature_lengths(self) -> List[int]:
        """Calculates the length of each feature

        Returns:
            A list of the lengths of each feature.
        """
        mol = MolFromSmiles('CC')
        return self._get_feature_lengths(mol)
