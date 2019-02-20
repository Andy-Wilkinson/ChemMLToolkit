import pytest
import chemmltoolkit.processing.smiles as smiles


class TestSmiles(object):
    @pytest.mark.parametrize("smiles_full,smiles_desalted", [
        ('CCCN.Cl', 'CCCN'),
        ('Cl.CCCN', 'CCCN'),
        ('[Na+].[O-]c1ccccc1', '[O-]c1ccccc1'),
        ('c1ccccc1[O-].[Na+]', 'c1ccccc1[O-]'),
    ])
    def test_desalt_smiles(self, smiles_full, smiles_desalted):
        smiles_result = smiles.desalt_smiles(smiles_full)
        assert smiles_result == smiles_desalted
