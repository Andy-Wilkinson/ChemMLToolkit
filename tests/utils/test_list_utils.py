import pytest
import chemmltoolkit.utils.list_utils as list_utils


class TestListUtils(object):
    @pytest.mark.parametrize("input,expected_result", [
        ([], []),
        ([42], [42]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([[1, 2], 3, [4, 5]], [1, 2, 3, 4, 5]),
        ([1, [2, [3, [4]]], [5]], [1, 2, 3, 4, 5]),
        (['ABC', 'D', 'EFG'], ['ABC', 'D', 'EFG']),
        (['AB', ['CD', ['EF'], 'GH']], ['AB', 'CD', 'EF', 'GH']),
    ])
    def test_flatten(self, input, expected_result):
        result = list_utils.flatten(input)
        assert result == expected_result

    @pytest.mark.parametrize("input,expected_result", [
        ({}, {}),
        ({'alpha': 42, 'beta': 'xyz'},
         {'alpha': 42, 'beta': 'xyz'}),
        ({'alpha': 42, 'inner': {'beta': 20, 'gamma': 'xyz'}},
         {'alpha': 42, 'inner.beta': 20, 'inner.gamma': 'xyz'})
    ])
    def test_flatten_dict(self, input, expected_result):
        result = list_utils.flatten_dict(input)
        assert result == expected_result

    @pytest.mark.parametrize("input,tokens,expected_result", [
        ('A', ['A', 'B', 'C', 'D'], [1, 0, 0, 0]),
        ('C', ['A', 'B', 'C', 'D'], [0, 0, 1, 0]),
        ('X', ['A', 'B', 'C', 'D'], [0, 0, 0, 0]),
    ])
    def test_one_hot(self, input, tokens, expected_result):
        result = list_utils.one_hot(input, tokens)
        assert result == expected_result

    @pytest.mark.parametrize("input,length,item,expected_result", [
        ([1, 2, 3], 5, 0, [1, 2, 3, 0, 0]),
        ([1, 2, 3], 5, None, [1, 2, 3, None, None]),
        ([1, 2, 3], 5, 'X', [1, 2, 3, 'X', 'X']),
        ([1, 2, 3, 4, 5], 5, 0, [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5, 6], 5, 0, [1, 2, 3, 4, 5, 6]),
    ])
    def test_pad_list(self, input, length, item, expected_result):
        result = list_utils.pad_list(input, length, item)
        assert result == expected_result

    @pytest.mark.parametrize("input,expected_result", [
        ([[1, 2, 3], ['A', 'B', 'C']], [(1, 'A'), (2, 'B'), (3, 'C')]),
        ([[1, 2, 3], 'X'], [(1, 'X'), (2, 'X'), (3, 'X')]),
        ([[1, 2, 3], 'ABC'], [(1, 'ABC'), (2, 'ABC'), (3, 'ABC')]),
        ([1, ['A', 'B', 'C']], [(1, 'A'), (1, 'B'), (1, 'C')]),
        ([[1, 2, 3], 9, [4, 5, 6]], [(1, 9, 4), (2, 9, 5), (3, 9, 6)]),
    ])
    def test_zip_expand(self, input, expected_result):
        result = list_utils.zip_expand(*input)
        result = [x for x in result]
        assert result == expected_result
