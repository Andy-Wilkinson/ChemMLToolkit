import pytest
from chemmltoolkit.processing.mols import MolCleaner
from rdkit import Chem


class TestMols_MolCleaner(object):
    @pytest.mark.parametrize("smiles_input,smiles_clean", [
        # Canonicalisation
        ('C1=CC=CC=C1', 'c1ccccc1'),  # Aromatics
        # Desalting
        ('CCCN.Cl', 'CCCN'),  # Remove common salts
        ('[Na+].[O-]c1ccccc1', 'Oc1ccccc1'),  # Remove common salts
        ('[O-]c1ccccc1.[Na+]', 'Oc1ccccc1'),  # Remove common salts
        ('CC(=O)O.CCN', 'CCN'),  # Remove common salts (even if smaller)
        ('CN(C)C.Cl.Cl.Br', 'CN(C)C'),  # Remove multiple common salts
        ('Oc1ccccc1.CCCN', 'Oc1ccccc1'),  # Remove uncommon salts
        ('CC(=O)O.[Na]', 'CC(=O)O'),  # Do not remove all salt components
        ('[Na].CC(=O)O', 'CC(=O)O'),  # Do not remove all salt components
        ('CC(=O)O.CC(=O)O', 'CC(=O)O'),  # Do not remove all salt components
        ('C1.C1.Cl', 'CC'),  # Joined fragments
        # Normalise Charges
        ('[O-]c1ccccc1', 'Oc1ccccc1'),  # Remove negative charge on O
        ('CC(=O)[O-]', 'CC(=O)O'),  # Remove negative charge on O
        ('C[NH+](C)(C)', 'CN(C)C'),  # Remove positive charge on N
        ('C[N+](C)(C)(C)', 'C[N+](C)(C)C'),  # Do not remove quaternary N+
        ('C[N+](C)=C\\C=C\\[O-]', 'CN(C)C=CC=O'),  # Multiple charges
        # Normalise functional groups (RDKit built in)
        ('CC[N](=O)=O', 'CC[N+](=O)[O-]'),  # Nitro
        ('CC[S+2]([O-])([O-])C', 'CCS(C)(=O)=O'),  # Sulfone
        ('c1ccn(=O)cc1', '[O-][n+]1ccccc1'),  # Py N-oxide
        # Normalise tautomers (custom)
        ('OCCc1n[nH]nn1', 'OCCc1nnn[nH]1'),  # Tetrazoles (normalise)
        ('OCCc1[nH]nnn1', 'OCCc1nnn[nH]1'),  # Tetrazoles (keep)
        ('OCCc1nn(C)nn1', 'Cn1nnc(CCO)n1'),  # Me-tetrazoles (keep)
        ('OCCc1n(C)nnn1', 'Cn1nnnc1CCO'),  # Me-tetrazoles (keep)
    ])
    def test_clean_mol(self, smiles_input, smiles_clean):
        mol_input = Chem.MolFromSmiles(smiles_input)
        mol_result = MolCleaner().clean_mol(mol_input)
        smiles_result = Chem.MolToSmiles(mol_result)
        assert smiles_result == smiles_clean

    @pytest.mark.parametrize("smiles_input,smiles_clean", [
        # Canonicalisation
        ('C1=CC=CC=C1', 'c1ccccc1'),  # Aromatics
        # Desalting
        ('CCCN.Cl', 'CCCN'),  # Remove common salts
        ('[Na+].[O-]c1ccccc1', 'Oc1ccccc1'),  # Remove common salts
        ('[O-]c1ccccc1.[Na+]', 'Oc1ccccc1'),  # Remove common salts
        ('CC(=O)O.CCN', 'CCN'),  # Remove common salts (even if smaller)
        ('CN(C)C.Cl.Cl.Br', 'CN(C)C'),  # Remove multiple common salts
        ('Oc1ccccc1.CCCN', 'Oc1ccccc1'),  # Remove uncommon salts
        ('CC(=O)O.[Na]', 'CC(=O)O'),  # Do not remove all salt components
        ('[Na].CC(=O)O', 'CC(=O)O'),  # Do not remove all salt components
        ('CC(=O)O.CC(=O)O', 'CC(=O)O'),  # Do not remove all salt components
        ('C1.C1.Cl', 'CC'),  # Joined fragments
        # Normalise Charges
        ('[O-]c1ccccc1', 'Oc1ccccc1'),  # Remove negative charge on O
        ('CC(=O)[O-]', 'CC(=O)O'),  # Remove negative charge on O
        ('C[NH+](C)(C)', 'CN(C)C'),  # Remove positive charge on N
        ('C[N+](C)(C)(C)', 'C[N+](C)(C)C'),  # Do not remove quaternary N+
        ('C[N+](C)=C\\C=C\\[O-]', 'CN(C)C=CC=O'),  # Multiple charges
        # Normalise functional groups (RDKit built in)
        ('CC[N](=O)=O', 'CC[N+](=O)[O-]'),  # Nitro
        ('CC[S+2]([O-])([O-])C', 'CCS(C)(=O)=O'),  # Sulfone
        ('c1ccn(=O)cc1', '[O-][n+]1ccccc1'),  # Py N-oxide
        # Normalise tautomers (custom)
        ('OCCc1n[nH]nn1', 'OCCc1nnn[nH]1'),  # Tetrazoles (normalise)
        ('OCCc1[nH]nnn1', 'OCCc1nnn[nH]1'),  # Tetrazoles (keep)
        ('OCCc1nn(C)nn1', 'Cn1nnc(CCO)n1'),  # Me-tetrazoles (keep)
        ('OCCc1n(C)nnn1', 'Cn1nnnc1CCO'),  # Me-tetrazoles (keep)
        ('c1ccc2c(c1)[i+]c3ccccc23', None),  # RDKit fails to read
    ])
    def test_clean_smiles(self, smiles_input, smiles_clean):
        smiles_result = MolCleaner().clean_smiles(smiles_input)
        assert smiles_result == smiles_clean
