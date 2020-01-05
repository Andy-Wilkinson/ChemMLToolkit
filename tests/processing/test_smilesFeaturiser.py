import pytest
from chemmltoolkit.processing import SmilesFeaturiser


class TestSmilesFeaturiser(object):
    @pytest.mark.parametrize("token_input,feature_names,expected_output", [
        # Tests for individual features
        ('C', ['token_index'], [0]),
        ('Br', ['token_index'], [2]),
        ('-', ['token_index'], [3]),
        ('C', ['token_onehot'], [1, 0, 0, 0]),
        ('Br', ['token_onehot'], [0, 0, 1, 0]),
        ('-', ['token_onehot'], [0, 0, 0, 1]),

        ('C', ['symbol_index'], [1]),
        ('n', ['symbol_index'], [2]),
        ('Br', ['symbol_index'], [3]),
        ('[Pt]', ['symbol_index'], [4]),
        ('-', ['symbol_index'], [0]),
        ('C', ['symbol_onehot'], [1, 0, 0, 0]),
        ('n', ['symbol_onehot'], [0, 1, 0, 0]),
        ('Br', ['symbol_onehot'], [0, 0, 1, 0]),
        ('[Pt]', ['symbol_onehot'], [0, 0, 0, 1]),
        ('-', ['symbol_onehot'], [0, 0, 0, 0]),

        # # Tests for multiple features
        # ('CCO', ['atomic_number', 'symbol_onehot', 'degree'], [
        #     [6, 0, 1, 0, 0, 1],
        #     [6, 0, 1, 0, 0, 2],
        #     [8, 0, 0, 1, 0, 1]]),
    ])
    def test_process_token(self,
                           token_input,
                           feature_names,
                           expected_output):
        featuriser = SmilesFeaturiser(feature_names,
                                      tokens_smiles=['C', 'N', 'Br', '-'],
                                      tokens_symbol=['C', 'N', 'Br', 'Pt'])
        features = featuriser.process_token(token_input)
        assert features == expected_output
        assert featuriser.get_feature_length() == len(expected_output)
