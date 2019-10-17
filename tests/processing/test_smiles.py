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
                                           token_unknown=placeholder)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens
        assert tokeniser.missing_tokens == {'S', 'N'}

    @pytest.mark.parametrize("smiles_str,token_sos,token_eos,tokens", [
        ('OCCN', None, None, [2, 0, 0, 1]),  # No SOS/EOS tokens
        ('OCCN', '<SOS>', None, [5, 2, 0, 0, 1]),  # SOS token
        ('OCCN', None, '<EOS>', [2, 0, 0, 1, 6]),  # EOS token
        ('OCCN', '<SOS>', '<EOS>', [5, 2, 0, 0, 1, 6]),  # SOS+EOS tokens
    ])
    def test_tokenise_smiles_with_sos_eos(self, smiles_str,
                                          token_sos, token_eos, tokens):
        token_list = ['C', 'N', 'O', 'Cl', 'Br', '<SOS>', '<EOS>']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           token_sos=token_sos,
                                           token_eos=token_eos)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

    @pytest.mark.parametrize("smiles_str,token_pad,length,truncate,tokens", [
        ('OCCN', None, None, False, [2, 0, 0, 1]),
        ('OCCN', '_', 10, False, [2, 0, 0, 1, 5, 5, 5, 5, 5, 5]),
        ('OCCN', '_', 10, True, [2, 0, 0, 1, 5, 5, 5, 5, 5, 5]),
        ('OCCN', ' ', 10, False, [2, 0, 0, 1, 6, 6, 6, 6, 6, 6]),
        ('OCCN', ' ', 10, True, [2, 0, 0, 1, 6, 6, 6, 6, 6, 6]),
        ('OCCN', ' ', 3, False, [2, 0, 0, 1]),
        ('OCCN', ' ', 3, True, [2, 0, 0]),
    ])
    def test_tokenise_smiles_with_padding(self, smiles_str,
                                          token_pad, length, truncate, tokens):
        token_list = ['C', 'N', 'O', 'Cl', 'Br', '_', ' ']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           token_padding=token_pad,
                                           sequence_length=length,
                                           truncate_sequence=truncate)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

    @pytest.mark.parametrize("smiles_str,token_sos,token_eos,tokens", [
        ('OCCN', None, None, [2, 0, 0, 1, 7, 7, 7]),  # No SOS/EOS tokens
        ('OCCN', '<SOS>', None, [5, 2, 0, 0, 1, 7, 7]),  # SOS token
        ('OCCN', None, '<EOS>', [2, 0, 0, 1, 6, 7, 7]),  # EOS token
        ('OCCN', '<SOS>', '<EOS>', [5, 2, 0, 0, 1, 6, 7]),  # SOS+EOS tokens
    ])
    def test_tokenise_smiles_with_pad_sos_eos(self, smiles_str,
                                              token_sos, token_eos, tokens):
        token_list = ['C', 'N', 'O', 'Cl', 'Br', '<SOS>', '<EOS>', '_']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           token_sos=token_sos,
                                           token_eos=token_eos,
                                           token_padding='_',
                                           sequence_length=7)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

    @pytest.mark.parametrize("smiles_str,split,tokens", [
        ('C3CC3', 'all_tokens', [1, 2, 1, 1, 2]),
        ('C%12CC%12', 'all_tokens', [1, 2, 1, 1, 2]),
        ('C1CC1C2CC2', 'all_tokens', [1, 2, 1, 1, 2, 1, 2, 1, 1, 2]),
        ('C1CC1C%12CC%12', 'all_tokens', [1, 2, 1, 1, 2, 1, 2, 1, 1, 2]),
        ('C1CC2C1CC2', 'all_tokens', [1, 2, 1, 1, 3, 1, 2, 1, 1, 3]),
        ('C1CC%12C1CC%12', 'all_tokens', [1, 2, 1, 1, 3, 1, 2, 1, 1, 3]),

        ('C3CC3', 'characters', [1, 2, 1, 1, 2]),
        ('C%12CC%12', 'characters', [1, 2, 1, 1, 2]),
        ('C1CC1C2CC2', 'characters', [1, 2, 1, 1, 2, 1, 2, 1, 1, 2]),
        ('C1CC1C%12CC%12', 'characters', [1, 2, 1, 1, 2, 1, 2, 1, 1, 2]),
        ('C1CC2C1CC2', 'characters', [1, 2, 1, 1, 3, 1, 2, 1, 1, 3]),
        ('C1CC%12C1CC%12', 'characters', [1, 2, 1, 1, 3, 1, 2, 1, 1, 3]),
    ])
    def test_tokenise_smiles_with_simplify_rings(self, smiles_str, split,
                                                 tokens):
        token_list = ['?', 'C', '1', '2', '3']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           splitting_method=split,
                                           simplify_rings=True)
        tokens_result = tokeniser.tokenise_smiles(smiles_str)
        assert tokens_result == tokens

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

    @pytest.mark.parametrize("smiles_str,split,tokens", [
        ('C1CC1', 'all_tokens', [1, 2, 1, 1, 2]),
        ('C1CC1', 'all_tokens', [1, 4, 1, 1, 4]),
        ('C1CC1C2CC2', 'all_tokens', [1, 2, 1, 1, 2, 1, 2, 1, 1, 2]),
        ('C1CC1C2CC2', 'all_tokens', [1, 2, 1, 1, 2, 1, 4, 1, 1, 4]),
        ('C1CC2C1CC2', 'all_tokens', [1, 2, 1, 1, 3, 1, 2, 1, 1, 3]),
        ('C1CC2C1CC2', 'all_tokens', [1, 2, 1, 1, 4, 1, 2, 1, 1, 4]),
        ('112233445566778899%10%10%11%11', 'all_tokens', [2] * 22),

        ('C1CC1', 'characters', [1, 2, 1, 1, 2]),
        ('C1CC1', 'characters', [1, 5, 2, 3, 1, 1, 5, 2, 3]),
        ('C1CC1C2CC2', 'characters', [1, 2, 1, 1, 2, 1, 2, 1, 1, 2]),
        ('C1CC1C2CC2', 'characters',
            [1, 2, 1, 1, 2, 1, 5, 2, 3, 1, 1, 5, 2, 3]),
        ('C1CC2C1CC2', 'characters', [1, 2, 1, 1, 3, 1, 2, 1, 1, 3]),
        ('C1CC2C1CC2', 'characters',
            [1, 2, 1, 1, 5, 2, 3, 1, 2, 1, 1, 5, 2, 3]),
        ('112233445566778899%10%10%11%11', 'characters', [2] * 22),
    ])
    def test_untokenise_smiles_with_simplify_rings(self, smiles_str, split,
                                                   tokens):
        token_list = ['?', 'C', '1', '2', '%12', '%']

        tokeniser = smiles.SmilesTokeniser(token_list,
                                           splitting_method=split,
                                           simplify_rings=True)
        smiles_result = tokeniser.untokenise_smiles(tokens)
        assert smiles_result == smiles_str

    def test_exception_if_padding_not_seq_length(self):
        token_list = ['?', 'C', '1', '2', '%12', '%']

        with pytest.raises(ValueError):
            _ = smiles.SmilesTokeniser(token_list, token_padding='?')

    def test_exception_if_seq_length_not_padding(self):
        token_list = ['?', 'C', '1', '2', '%12', '%']

        with pytest.raises(ValueError):
            _ = smiles.SmilesTokeniser(token_list, sequence_length=50)
