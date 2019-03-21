def desalt_smiles(smiles: str) -> str:
    """Removes all salts from the specified SMILES strings.

    This is a very simple and quick desalting algorithm that returns
    the longest component withing the input string. This will fail
    for cases with embedded components (e.g. 'c1cc([O-].[Na+])cccc1')
    or where individual components are actually connected
    (e.g. ethane represented as 'C1.C1').

    Args:
        smiles: The input SMILES string.

    Returns:
        The desalted SMILES string.
    """
    components = smiles.split('.')
    return max(components, key=len)


def split_smiles(smiles: str) -> list:
    """Splits a SMILES string into individual tokens.

    This will split a SMILES string into tokens representing
    individual parts of the SMILES string. Each individual token
    will be one of the following,

    - A single atom (including hydrogens or charge if specified)
    - A ring connection number (single digit or multi-digit)
    - Another SMILES character

    Args:
        smiles: The input SMILES string.

    Returns:
        A list of tokens.
    """
    result = []
    while smiles != '':
        if smiles[:2] == 'Cl' or smiles[:2] == 'Br':
            token_length = 2
        elif smiles[0] == '%':  # Ring linkage numbers >9 are prefixed by '%'
            token_length = 3
        elif smiles[0] == '[':
            token_length = smiles.index(']') + 1
        else:
            token_length = 1

        result.append(smiles[:token_length])
        smiles = smiles[token_length:]
    return result


def _split_smiles_halogen_only(smiles: str) -> list:
    result = []
    while smiles != '':
        if smiles[:2] == 'Cl' or smiles[:2] == 'Br':
            token_length = 2
        else:
            token_length = 1

        result.append(smiles[:token_length])
        smiles = smiles[token_length:]
    return result


def get_token_list(max_ring_number=9, chiral_carbon=True) -> list:
    tokens = []

    # SMILES 'organic' subset that do not need '[...]'
    tokens += ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
    tokens += ['c', 'n', 'o', 's']
    tokens += ['[nH]', '[n+]', '[N+]', '[O-]']

    # Chiral atoms
    if chiral_carbon:
        tokens += ['[C@]', '[C@@]', '[C@H]', '[C@@H]']

    # Control tokens
    tokens += [
        '-', '=', '#',  # Bonds
        '(', ')',  # Atom groups
        '/', '\\',  # Double bond geometry
    ]

    # Ring tokens
    tokens += [f'{x}' for x in range(1, min(max_ring_number, 9) + 1)] \
        + [f'%{x}' for x in range(10, max_ring_number + 1)]

    return tokens


class SmilesTokeniser(object):
    def __init__(self,
                 tokens: list,
                 splitting_method: str = 'all_tokens',
                 unknown_placeholder: str = None):
        """Tokeniser to convert SMILES strings to and from tokenised format.

        Before tokenising, the SMILES string is split into its constituent
        components. This can be performed in a number of ways
            - 'all_tokens' will correctly split all SMILES tokens
            - 'halogens_only' is character based apart from Cl and Br
            - 'characters' is entirely character based

        Args:
            tokens: A list of strings containing the token list to use.
            splitting_method: The technique for splitting the SMILES string.
            unknown_placeholder: The token to use for unidentified tokens, or
                None if SMILES string with unknown tokens should be rejected.
        """
        self._tokens_list = tokens
        self._tokens_missing = set()
        self._token_unknown = tokens.index(unknown_placeholder)\
            if unknown_placeholder else None
        self._token_lookup = {t: i for i, t in enumerate(tokens)}

        if splitting_method == 'all_tokens':
            self._split_function = split_smiles
        elif splitting_method == 'halogens_only':
            self._split_function = _split_smiles_halogen_only
        elif splitting_method == 'characters':
            self._split_function = list

    def tokenise_smiles(self, smiles: str) -> list:
        """Tokenises the specified SMILES string.

        This function will take a SMILES string and return the corresponding
        tokens as a list of integer values. The processing used is
        specified when creating this SmilesTokeniser.

        Args:
            smiles: The SMILES string to tokenise.
            unknown_placeholder: The token to use for unidentified tokens, or
                None if SMILES string with unknown tokens should be rejected.

        Returns:
            A list of integer tokens.
        """
        components = self._split_function(smiles)
        smiles_tokens = [self._get_token(c) for c in components]

        if None in smiles_tokens:
            return None
        else:
            return smiles_tokens

    def untokenise_smiles(self, smiles_tokens: list) -> str:
        """Returns the SMILES string corresponding to the tokenised representation.

        This function will take a list of tokens and will return the decoded
        SMILES string. The processing used is
        specified when creating this SmilesTokeniser.

        Args:
            smiles_tokens: A list of integer tokens.

        Returns:
            The corresponding SMILES string.
        """
        chars = [self._tokens_list[t] for t in smiles_tokens]
        return ''.join(chars)

    def _get_token(self, val):
        token = self._token_lookup.get(val)
        if token is not None:
            return token
        else:
            self._tokens_missing.add(val)
            return self._token_unknown
