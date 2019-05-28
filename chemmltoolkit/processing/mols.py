from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles, SanitizeMol
from rdkit.Chem.AllChem import ReactionFromSmarts
from rdkit.Chem.MolStandardize import rdMolStandardize


class MolCleaner(object):
    def __init__(self):
        self._tautomerTetrazole = ReactionFromSmarts(
            '[*:1]-c1n[nH]nn1>>[*:1]-c1[nH]nnn1')

    def clean_mol(self, mol: Mol) -> Mol:
        """Cleans the specified molecule into standardised format.

        The steps are,

        - Removal of salts
        - Normalise structures
        - Normalise tautomers
        - Remove all charges (where possible)

        Args:
            mol: The molecule to clean.

        Returns:
            The cleaned molecule.
        """
        # Use RDKit standardizer to return the parent fragment (non-salt)
        # This will also apply more normalisation and clean up any charges

        mol = rdMolStandardize.ChargeParent(mol)

        # Custom tautomers

        mol = self._apply_reaction(mol, self._tautomerTetrazole)

        return mol

    def clean_smiles(self, smiles: str) -> str:
        """Cleans the specified SMILES into standardised format.

        The steps are,

        - Removal of salts
        - Normalise structures
        - Normalise tautomers
        - Remove all charges (where possible)

        Args:
            smiles: The SMILES to clean.

        Returns:
            The cleaned SMILES.
        """
        mol_input = MolFromSmiles(smiles)
        if not mol_input:
            return None
        mol_clean = self.clean_mol(mol_input)
        return MolToSmiles(mol_clean)

    def _apply_reaction(self, mol, reaction):
        products = reaction.RunReactants((mol,))
        if products:
            mol = products[0][0]
            SanitizeMol(mol)
            return mol
        else:
            return mol
