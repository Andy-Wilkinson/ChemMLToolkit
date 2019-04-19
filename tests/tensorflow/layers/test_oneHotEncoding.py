import pytest
from chemmltoolkit.tensorflow.layers import OneHotEncoding


class TestOneHotEncoding(object):
    @pytest.mark.parametrize("input,depth,expected_output", [
        ([2, 1, 3], 4, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]),
    ])
    def test_call(self, input, depth, expected_output):
        oneHotEncoding = OneHotEncoding(depth)
        output = oneHotEncoding(input)
        assert output == expected_output
