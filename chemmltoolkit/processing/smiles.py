from chemmltoolkit.utils.list_utils import pad_list


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
    """Tokeniser to convert SMILES strings to and from tokenised format.

    Before tokenising, the SMILES string is split into its constituent
    components. This can be performed in a number of ways
        - 'all_tokens' will correctly split all SMILES tokens
        - 'halogens_only' is character based apart from Cl and Br
        - 'characters' is entirely character based

    Args:
        tokens: A list of strings containing the token list to use.
        splitting_method: The technique for splitting the SMILES string.
        token_unknown: The token to use for unidentified tokens, or
            None if SMILES string with unknown tokens should be rejected.
        token_sos: A token to add at the start of the sequence.
        token_eos: A token to add at the end of the sequence.
        token_padding: A token to use to pad to the desired length.
        sequence_length: The desired length of the resulting sequence.
        truncate_sequence: Whether to truncate the sequence if it is longer
            than 'sequence_length'.
        simplify_rings: Will re-use ring number tokens where possible.
    """

    def __init__(self,
                 tokens: list,
                 splitting_method: str = 'all_tokens',
                 token_unknown: str = None,
                 token_sos: str = None,
                 token_eos: str = None,
                 token_padding: str = None,
                 sequence_length: int = None,
                 truncate_sequence: bool = False,
                 simplify_rings: bool = False):

        if (token_padding and not sequence_length) or \
           (sequence_length and not token_padding):
            raise ValueError('Both `token_padding` and `sequence_length`' +
                             'must be set together.')

        self._tokens_list = tokens
        self._tokens_missing = set()
        self._token_lookup = {t: i for i, t in enumerate(tokens)}
        self._token_unknown = tokens.index(token_unknown)\
            if token_unknown else None
        self._sequence_sos = [tokens.index(token_sos)]\
            if token_sos else []
        self._sequence_eos = [tokens.index(token_eos)]\
            if token_eos else []
        self._token_padding = tokens.index(token_padding)\
            if token_padding else None
        self._sequence_length = sequence_length
        self._truncate_sequence = truncate_sequence
        self._splitting_method = splitting_method
        self._simplify_rings = simplify_rings

        if splitting_method == 'all_tokens':
            self._split_function = split_smiles
        elif splitting_method == 'halogens_only':
            self._split_function = _split_smiles_halogen_only
        elif splitting_method == 'characters':
            self._split_function = list

    @property
    def missing_tokens(self):
        """Returns a set containing any SMILES tokens that have not been
        found in the supplied tokens list.
        """
        return self._tokens_missing

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

        # If we are not splitting by 'all_tokens' and ring numbering is greater
        # than 9 (i.e. %xx numbering) fallback to pre-simplifying ring with
        # 'all_tokens' first, then use the desired splitting method
        preprocess_ring_simplification = self._simplify_rings \
            and self._splitting_method != 'all_tokens' \
            and '%' in smiles

        if preprocess_ring_simplification:
            components = split_smiles(smiles)
            components = self._ring_simplifier(components)
            smiles = ''.join(components)

        components = self._split_function(smiles)

        if self._simplify_rings and not preprocess_ring_simplification:
            components = self._ring_simplifier(components)

        smiles_tokens = [self._get_token(c) for c in components]

        if None in smiles_tokens:
            return None

        smiles_tokens = self._sequence_sos + smiles_tokens + self._sequence_eos

        if self._token_padding:
            smiles_tokens = pad_list(smiles_tokens,
                                     self._sequence_length,
                                     self._token_padding)

            if self._truncate_sequence:
                smiles_tokens = smiles_tokens[:self._sequence_length]

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
        components = [self._tokens_list[t] for t in smiles_tokens]

        if self._simplify_rings:
            # If we aren't using 'all_tokens' then swap to this format
            if self._splitting_method != 'all_tokens':
                components = split_smiles(''.join(components))
            components = self._ring_unsimplifier(components)

        return ''.join(components)

    def _get_token(self, val):
        token = self._token_lookup.get(val)
        if token is not None:
            return token
        else:
            self._tokens_missing.add(val)
            return self._token_unknown

    def _ring_simplifier(self, components):
        ring_rename_dictionary = {}
        num_tokens = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '%']

        def get_next_id():
            for id in range(1, 100):
                id_str = str(id) if id < 10 else '%' + str(id)
                if id_str not in ring_rename_dictionary:
                    return id_str

        def simplify_component(component):
            if component[0] in num_tokens:
                if component not in ring_rename_dictionary:
                    new_id = get_next_id()
                    ring_rename_dictionary[component] = new_id
                    return new_id
                else:
                    return ring_rename_dictionary.pop(component)
            else:
                return component

        return [simplify_component(c) for c in components]

    def _ring_unsimplifier(self, components):
        ring_rename_dictionary = {}
        num_tokens = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '%']
        next_id = 1

        def unsimplify_component(component):
            nonlocal next_id
            if component[0] in num_tokens:
                if component not in ring_rename_dictionary:
                    id = str(next_id) if next_id < 10 else '%' + str(next_id)
                    next_id += 1
                    ring_rename_dictionary[component] = id
                    return id
                else:
                    return ring_rename_dictionary.pop(component)
            else:
                return component

        return [unsimplify_component(c) for c in components]
