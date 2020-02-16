from chemmltoolkit.utils.list_utils import flatten
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol


class MoleculeFeaturiser:
    """Generator for molecule-based features.

    Args:
        features: A list of features to generate.
    """
    def __init__(self,
                 features: list):
        self.features = features

    def process_molecule(self, mol: Mol):
        """Generates molecular features.

        Args:
            mol: The molecule to featurise.

        Returns:
            A list of features.
        """

        features = [feature(mol) for feature in self.features]
        return flatten(features)

    def get_feature_length(self) -> int:
        """Calculates the length of the generated feature vector

        Returns:
            The length of the feature vector.
        """
        molecule = MolFromSmiles('CC')
        features = self.process_molecule(molecule)
        return len(features)
