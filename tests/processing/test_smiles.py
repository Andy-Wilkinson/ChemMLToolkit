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


class TestSmilesSmilesTokeniser(object):
    @pytest.mark.parametrize("smiles_str,tokens", [
        ('CCCN', [1, 1, 1, 2]),  # Simple example
        ('C=C', [1, 9, 1]),  # Bond characters
        ('CN(C)C', [1, 2, 7, 1, 8, 1]),  # Braces
        ('ClCCBr', [4, 1, 1, 5]),  # Two-character atoms
        ('O[Pt]O', [3, 6, 3]),  # [xxx] atoms
        ('C1CC1', [1, 10, 1, 1, 10]),  # Ring numbering (<=9)
        ('C%10CC%10', [1, 13, 1, 1, 13]),  # Ring numbering (>9)
    ])
    def test_tokenise_smiles_all_tokens(self, smiles_str, tokens):
        token_list = ['?', 'C', 'N', 'O', 'Cl', 'Br', '[Pt]',
                      '(', ')', '=', '1', '2', '3', '%10']

        tokeniser = smiles.SmilesTokeniser(token_list)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

    @pytest.mark.parametrize("smiles_str,tokens", [
        ('CCCN', [1, 1, 1, 2]),  # Simple example
        ('C=C', [1, 12, 1]),  # Bond characters
        ('CN(C)C', [1, 2, 10, 1, 11, 1]),  # Braces
        ('ClCCBr', [4, 1, 1, 5]),  # Two-character atoms
        ('O[Pt]O', [3, 6, 7, 8, 9, 3]),  # [xxx] atoms
        ('C1CC1', [1, 13, 1, 1, 13]),  # Ring numbering (<=9)
        ('C%10CC%10', [1, 16, 13, 17, 1, 1, 16, 13, 17]),  # Ring number (>9)
    ])
    def test_tokenise_smiles_halogens_only(self, smiles_str, tokens):
        token_list = ['?', 'C', 'N', 'O', 'Cl', 'Br', '[', 'P', 't', ']',
                      '(', ')', '=', '1', '2', '3', '%', '0']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           splitting_method='halogens_only')
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

    @pytest.mark.parametrize("smiles_str,tokens", [
        ('CCCN', [1, 1, 1, 2]),  # Simple example
        ('C=C', [1, 12, 1]),  # Bond characters
        ('CN(C)C', [1, 2, 10, 1, 11, 1]),  # Braces
        ('ClCCBr', [1, 4, 1, 1, 5, 18]),  # Two-character atoms
        ('O[Pt]O', [3, 6, 7, 8, 9, 3]),  # [xxx] atoms
        ('C1CC1', [1, 13, 1, 1, 13]),  # Ring numbering (<=9)
        ('C%10CC%10', [1, 16, 13, 17, 1, 1, 16, 13, 17]),  # Ring number (>9)
    ])
    def test_tokenise_smiles_characters(self, smiles_str, tokens):
        token_list = ['?', 'C', 'N', 'O', 'l', 'B', '[', 'P', 't', ']',
                      '(', ')', '=', '1', '2', '3', '%', '0', 'r']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           splitting_method='characters')
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

    @pytest.mark.parametrize("smiles_str,split,placeholder,tokens", [
        ('SBr[Pt]NC', 'all_tokens', '?', [4, 1, 5, 4, 0]),
        ('SBr[Pt]NC', 'all_tokens', None, None),
        ('SBr[Pt]NC', 'halogens_only', '?', [4, 1, 6, 7, 8, 9, 4, 0]),
        ('SBr[Pt]NC', 'halogens_only', None, None),
        ('SBr[Pt]NC', 'characters', '?', [4, 2, 3, 6, 7, 8, 9, 4, 0]),
        ('SBr[Pt]NC', 'characters', None, None),
    ])
    def test_tokenise_smiles_with_unknown_placeholder(self, smiles_str, split,
                                                      placeholder, tokens):
        token_list = ['C', 'Br', 'B', 'r', '?', '[Pt]', '[', 'P', 't', ']']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           splitting_method=split,
                                           unknown_placeholder=placeholder)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens
        assert tokeniser.missing_tokens == {'S', 'N'}

    @pytest.mark.parametrize("smiles_str,tokens", [
        ('CCCN', [1, 1, 1, 2]),  # Simple example
        ('C=C', [1, 9, 1]),  # Bond characters
        ('CN(C)C', [1, 2, 7, 1, 8, 1]),  # Braces
        ('ClCCBr', [4, 1, 1, 5]),  # Two-character atoms
        ('O[Pt]O', [3, 6, 3]),  # [xxx] atoms
        ('C1CC1', [1, 10, 1, 1, 10]),  # Ring numbering (<=9)
        ('C%10CC%10', [1, 13, 1, 1, 13]),  # Ring numbering (>9)
        ('?=C=?', [0, 9, 1, 9, 0]),  # Unknown tokens
    ])
    def test_untokenise_smiles(self, smiles_str, tokens):
        token_list = ['?', 'C', 'N', 'O', 'Cl', 'Br', '[Pt]',
                      '(', ')', '=', '1', '2', '3', '%10']

        tokeniser = smiles.SmilesTokeniser(token_list)
        smiles_result = tokeniser.untokenise_smiles(tokens)
        assert smiles_result == smiles_str
