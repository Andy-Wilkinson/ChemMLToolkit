import pytest
import numpy as np
import chemmltoolkit.processing.adjacency_ops as adjacency_ops


class TestAdjacencyOps(object):
    @pytest.mark.parametrize("input,expected_result", [
        ([
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
        ], [
            [[0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0]],
            [[1, 2, 3, 1], [2, 3, 4, 1], [3, 4, 5, 1], [1, 1, 1, 0]],
        ]),
    ])
    def test_add_master_node(self, input, expected_result):
        input = np.array(input)
        reference_input = input.copy()
        result = adjacency_ops.add_master_node(input)
        assert (result == expected_result).all()
        assert (input == reference_input).all()

    @pytest.mark.parametrize("input,expected_result", [
        ([
            [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
        ], [
            [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
            [[1, 2, 3], [2, 1, 4], [3, 4, 1]],
        ]),
    ])
    def test_add_self_loops(self, input, expected_result):
        input = np.array(input)
        reference_input = input.copy()
        result = adjacency_ops.add_self_loops(input)
        assert (result == expected_result).all()
        assert (input == reference_input).all()

    @pytest.mark.parametrize("input,expected_result", [
        ([
            [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
            [[1, 2, 2], [2, 3, 5], [2, 5, 3]],
        ], [
            [[0, 0, 1], [0, 0, 1], [0.5, 0.5, 0]],
            [[0.2, 0.4, 0.4], [0.2, 0.3, 0.5], [0.2, 0.5, 0.3]],
        ]),
    ])
    def test_normalise(self, input, expected_result):
        input = np.array(input)
        reference_input = input.copy()
        result = adjacency_ops.normalise(input)
        result = np.round(result, 2)
        assert (result == expected_result).all()
        assert (input == reference_input).all()

    @pytest.mark.parametrize("input,expected_result", [
        ([
            [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
            [[1, 2, 2], [2, 3, 5], [2, 5, 3]],
        ], [
            [[0, 0, 0.71], [0, 0, 0.71], [0.71, 0.71, 0]],
            [[0.2, 0.28, 0.28], [0.28, 0.3, 0.5], [0.28, 0.5, 0.3]],
        ]),
    ])
    def test_normalise_spectral(self, input, expected_result):
        input = np.array(input)
        reference_input = input.copy()
        result = adjacency_ops.normalise_spectral(input)
        result = np.round(result, 2)
        assert (result == expected_result).all()
        assert (input == reference_input).all()
