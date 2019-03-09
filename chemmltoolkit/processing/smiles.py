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
