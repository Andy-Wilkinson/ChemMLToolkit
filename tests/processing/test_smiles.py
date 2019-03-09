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

    @pytest.mark.parametrize("smiles_input,smiles_tokens", [
        ('CCCN', ['C', 'C', 'C', 'N']),  # Simple example
        ('C=C', ['C', '=', 'C']),  # Bond characters
        ('CN(C)C', ['C', 'N', '(', 'C', ')', 'C']),  # Braces
        ('ClCCBr', ['Cl', 'C', 'C', 'Br']),  # Two-character atoms
        ('O[Pt]O', ['O', '[Pt]', 'O']),  # [xxx] atoms
        ('C1CC1', ['C', '1', 'C', 'C', '1']),  # Ring numbering (<=9)
        ('C%10CC%10', ['C', '%10', 'C', 'C', '%10']),  # Ring numbering (>9)
    ])
    def test_split_smiles(self, smiles_input, smiles_tokens):
        result = smiles.split_smiles(smiles_input)
        assert result == smiles_tokens

    tokens_atoms = ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I',
                    'c', 'n', 'o', 's', '[nH]', '[n+]', '[N+]', '[O-]']
    tokens_chiralC = ['[C@]', '[C@@]', '[C@H]', '[C@@H]']
    tokens_control = ['-', '=', '#', '(', ')', '/', '\\']

    def test_get_token_list(self):
        token_list = smiles.get_token_list()
        expected = self.tokens_atoms \
            + self.tokens_chiralC \
            + self.tokens_control \
            + ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        assert token_list == expected

    def test_get_token_list_ringcount0(self):
        token_list = smiles.get_token_list(max_ring_number=0)
        expected = self.tokens_atoms \
            + self.tokens_chiralC \
            + self.tokens_control
        assert token_list == expected

    def test_get_token_list_ringcount5(self):
        token_list = smiles.get_token_list(max_ring_number=5)
        expected = self.tokens_atoms \
            + self.tokens_chiralC \
            + self.tokens_control \
            + ['1', '2', '3', '4', '5']
        assert token_list == expected

    def test_get_token_list_ringcount12(self):
        token_list = smiles.get_token_list(max_ring_number=12)
        expected = self.tokens_atoms \
            + self.tokens_chiralC \
            + self.tokens_control \
            + ['1', '2', '3', '4', '5', '6', '7', '8', '9'] \
            + ['%10', '%11', '%12']
        assert token_list == expected

    def test_get_token_list_nochiralC(self):
        token_list = smiles.get_token_list(chiral_carbon=False)
        expected = self.tokens_atoms \
            + self.tokens_control \
            + ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        assert token_list == expected
