import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.layers import GraphConv


class TestGraphConv(tf.test.TestCase):
    # The same graph structure is used for most tests.
    # There are two examples per batch, each with three nodes.
    #
    # Example 1:
    #    Features:  a[1,4], b[3,1], c[2,6]
    #    Adjacency matrix 1:  a-c-b
    #    Adjacency matrix 2:  a-b-c
    #
    # Example 2:
    #    Features:  a[2,1], b[1,3], c[3,4]
    #    Adjacency matrix 1:  (edges between all nodes)
    #    Adjacency matrix 2:  b-a-c

    def test_call_single_adjacency(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            [[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]
        ]
        expected_output = [
            [[2.0, 6.0], [2.0, 6.0], [4.0, 5.0]],
            [[4.0, 7.0], [5.0, 5.0], [3.0, 4.0]]
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        kernel_initializer=initializer_identitiy_3d)

    def test_call_adjacency_concat(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            ],
            [
                [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ]
        ]
        expected_output = [
            [[2.0, 6.0, 3.0, 1.0],
             [2.0, 6.0, 3.0, 10.0],
             [4.0, 5.0, 3.0, 1.0]],
            [[4.0, 7.0, 4.0, 7.0],
             [5.0, 5.0, 2.0, 1.0],
             [3.0, 4.0, 2.0, 1.0]],
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        aggregation_method='concat',
                        kernel_initializer=initializer_identitiy_3d)

    def test_call_adjacency_sum(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            ],
            [
                [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ]
        ]
        expected_output = [
            [[5.0, 7.0], [5.0, 16.0], [7.0, 6.0]],
            [[8.0, 14.0], [7.0, 6.0], [5.0, 5.0]]
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        aggregation_method='sum',
                        kernel_initializer=initializer_identitiy_3d)

    def test_call_adjacency_max(self):
        input_features = [
            [[1.0, 4.0], [3.0, 1.0], [2.0, 6.0]],
            [[2.0, 1.0], [1.0, 3.0], [3.0, 4.0]]
        ]
        input_adjacency = [
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            ],
            [
                [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ]
        ]
        expected_output = [
            [[3.0, 6.0], [3.0, 10.0], [4.0, 5.0]],
            [[4.0, 7.0], [5.0, 5.0], [3.0, 4.0]]
        ]

        self._test_call([input_features, input_adjacency], expected_output,
                        units=2, use_bias=False,
                        aggregation_method='max',
                        kernel_initializer=initializer_identitiy_3d)

    def _test_call(self, input, expected_output, **kwargs):
        input = [np.array(i) for i in input]
        graphConv = GraphConv(**kwargs)
        computed_shape = graphConv.compute_output_shape(
            [input[0].shape, input[1].shape])
        output = graphConv(input)

        self.assertAllEqual(expected_output, output)
        self.assertAllEqual(output.shape, computed_shape)


def initializer_identitiy_3d(shape, dtype=None):
    identity = tf.eye(shape[1], shape[2], dtype=dtype)
    identity = tf.reshape(identity, (1, shape[1], shape[2]))
    return tf.tile(identity, (shape[0], 1, 1))
