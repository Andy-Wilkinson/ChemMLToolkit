from chemmltoolkit.utils.list_utils import flatten
from chemmltoolkit.utils.list_utils import one_hot


class SmilesFeaturiser:
    """Generator for a wide range of features from SMILES strings.

    A number of features are implemented,
        - token_index: The raw SMILES token index (int)
        - token_onehot: The raw SMILES token (one-hot)

    Args:
        features: A list of feature name strings to generate.
        tokens_smiles: List of tokens to use for raw SMILES tokens.
            (no default)
        tokens_symbol: List of tokens to use for atom symbols.
            (no default)
    """

    def __init__(self,
                 features: list,
                 tokens_smiles: list = None,
                 tokens_symbol: list = None):
        self.features = [self._get_feature(feature) for feature in features]
        self.tokens_smiles = tokens_smiles
        self.tokens_symbol = tokens_symbol

    def process_token(self, token: str):
        """Generates features for an individual SMILES token.

        Args:
            token: The SMILES token to featurise.

        Returns:
            A list of features.
        """

        features = [feature(token) for feature in self.features]
        return flatten(features)

    # def process_molecule(self, mol: Mol):
    #     """Generates features for all atoms in a molecule.

    #     Args:
    #         mol: The molecule to featurise.

    #     Returns:
    #         A nested list of features for all atoms.
    #     """
    #     atoms = mol.GetAtoms()
    #     return [self.process_atom(atom) for atom in atoms]

    def get_feature_length(self) -> int:
        """Calculates the length of the generated feature vector

        Returns:
            The length of the feature vector.
        """
        features = self.process_token('C')
        return len(features)

    def _get_feature(self, name):
        func_name = f'_f_{name}'
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise f'Undefined SMILES feature: {name}'

    def _parse_symbol(self, token: str):
        if token[0] >= 'A' and token[0] <= 'Z':
            return token
        elif token in ['c', 'n', 'o', 's']:
            return token.upper()
        elif token[0] == '[':
            return token[1:-1]
        else:
            return None

    def _f_token_index(self, token: str):
        return self.tokens_smiles.index(token)

    def _f_token_onehot(self, token: str):
        return one_hot(token, self.tokens_smiles)

    def _f_symbol_index(self, token: str):
        symbol = self._parse_symbol(token)
        if not symbol:
            return 0
        return self.tokens_symbol.index(symbol) + 1

    def _f_symbol_onehot(self, token: str):
        symbol = self._parse_symbol(token)
        return one_hot(symbol, self.tokens_symbol)
