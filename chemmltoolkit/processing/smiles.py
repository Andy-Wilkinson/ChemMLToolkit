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
