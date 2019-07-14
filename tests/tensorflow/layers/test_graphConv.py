import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.layers import GraphConv


class TestGraphConv(tf.test.TestCase):
    def test_call_simple(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ]
        expected_output = [
            [[2.0, 6.0], [2.0, 6.0], [4.0, 5.0]],
            [[4.0, 7.0], [5.0, 5.0], [3.0, 4.0]]
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        kernel_initializer='identity')

    def test_call_with_self_loops(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ]
        expected_output = [
            [[3.0, 10.0], [5.0, 7.0], [6.0, 11.0]],
            [[6.0, 8.0], [6.0, 8.0], [6.0, 8.0]]
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        add_adjacency_self_loops=True,
                        kernel_initializer='identity')

    def test_call_normalised_adjacency(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ]
        expected_output = [
            [[2.0, 6.0], [2.0, 6.0], [2.0, 2.5]],
            [[2.0, 3.5], [2.5, 2.5], [1.5, 2.0]]
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        normalize_adjacency_matrix=True,
                        kernel_initializer='identity')

    def test_preprocess_adjacency_self_loops(self):
        input_adjacency = [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ]
        expected_output = [
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ]

        output = GraphConv.preprocess_adjacency_matrices(
            input_adjacency,
            add_adjacency_self_loops=True)

        self.assertAllEqual(expected_output, output)

    def test_preprocess_adjacency_normalized(self):
        input_adjacency = [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ]
        expected_output = [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ]

        output = GraphConv.preprocess_adjacency_matrices(
            input_adjacency,
            normalize_adjacency_matrix=True)

        self.assertAllEqual(expected_output, output)

    def _test_call(self, input, expected_output, **kwargs):
        input = [np.array(i) for i in input]
        graphConv = GraphConv(**kwargs)
        computed_shape = graphConv.compute_output_shape(
            [input[0].shape, input[1].shape])
        output = graphConv(input)

        self.assertAllEqual(expected_output, output)
        self.assertAllEqual(output.shape, computed_shape)
