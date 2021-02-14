import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.layers import GraphConv
from chemmltoolkit.tensorflow.graph import TensorGraph
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.processing import adjacency_ops
from tests.test_utils.math_utils import Σ, Σneighbours
from tests.test_utils.math_utils import generate_random_node_features
from tests.test_utils.math_utils import generate_random_adjacency


class TestGraphConv(tf.test.TestCase):
    def test_implementation_relationalgcn(self):
        self._test_implementation_relationalgcn()

    def test_implementation_relationalgcn_withselfloops(self):
        self._test_implementation_relationalgcn(add_self_loops=True)

    def test_implementation_relationalgcn_sparseadjacency(self):
        self._test_implementation_relationalgcn(sparse_adjacency=True)

    def test_implementation_relationalgcn_sparseadjacency_withselfloops(self):
        self._test_implementation_relationalgcn(add_self_loops=True,
                                                sparse_adjacency=True)

    def _test_implementation_relationalgcn(self,
                                           add_self_loops=False,
                                           sparse_adjacency=False):
        # Literature reference for relational GCNs
        # "Modeling Relational Data with Graph Convolutional Networks"
        # (https://arxiv.org/abs/1703.06103)

        num_units = 8
        num_batches = 3
        num_nodes = 10
        num_node_features = 4
        num_edge_features = 5

        random = np.random.RandomState(seed=42)
        in_features = generate_random_node_features(random,
                                                    num_batches,
                                                    num_nodes,
                                                    num_node_features)
        in_adjacency = generate_random_adjacency(random,
                                                 num_batches,
                                                 num_nodes,
                                                 num_edge_features)

        # ChemMLToolkit implementation

        graphConv = GraphConv(num_units,
                              use_bias=False,
                              add_self_loops=add_self_loops)

        tensor_features = tf.convert_to_tensor(in_features, dtype=tf.float32)
        tensor_adjacency = tf.convert_to_tensor(in_adjacency, dtype=tf.float32)

        tensor_adjacency = adjacency_ops.normalise(tensor_adjacency)

        if sparse_adjacency:
            tensor_adjacency = tf.sparse.from_dense(tensor_adjacency)

        graph = TensorGraph(tensor_features, tensor_adjacency)

        graphConv.build({k: t.shape for k, t in graph.items()})
        graph_out = graphConv(graph)

        # Literature reference implementation
        # Equation 2 defines the RGCN layer

        W = np.transpose(graphConv.kernel.numpy(), (0, 2, 1))

        if add_self_loops:
            W0, W = W[0], W[1:]
        else:
            W0 = np.zeros((num_units, num_node_features))

        def activation(x): return np.maximum(x, 0)  # noqa: E731
        def c(adj, i, r): return np.sum(adj[r][i])  # noqa: E731

        def h_lplus1(i, h_l, adj): return (  # noqa: E731
            activation(Σ((0, num_edge_features), lambda r:
                         Σneighbours(adj[r], i, lambda j:
                                     (1.0/c(adj, i, r))
                                     * np.matmul(W[r], h_l[j])))
                       + np.matmul(W0, h_l[i])))

        np_out = [[h_lplus1(atom_idx, in_features[batch], in_adjacency[batch])
                   for atom_idx in range(num_nodes)]
                  for batch in range(num_batches)]

        # Check equivalence

        self.assertAllClose(graph_out[NODE_FEATURES], np_out)

    # The same graph structure is used for most of the following tests.
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
                        kernel_initializer=initializer_identity_3d)

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
                        kernel_initializer=initializer_identity_3d)

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
                        kernel_initializer=initializer_identity_3d)

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
                        kernel_initializer=initializer_identity_3d)

    def _test_call(self, input, expected_output, **kwargs):
        input = [tf.convert_to_tensor(np.array(i), dtype=tf.float32)
                 for i in input]
        graph = TensorGraph(input[0], input[1])
        graphConv = GraphConv(**kwargs)
        computed_shape = graphConv.compute_output_shape(
            {k: t.shape for k, t in graph.items()})
        graph_out = graphConv(graph)

        self.assertAllEqual(expected_output, graph_out[NODE_FEATURES])
        self.assertAllEqual({k: t.shape for k, t in graph_out.items()},
                            computed_shape)


def initializer_identity_3d(shape, dtype=None):
    identity = tf.eye(shape[1], shape[2], dtype=dtype)
    identity = tf.reshape(identity, (1, shape[1], shape[2]))
    return tf.tile(identity, (shape[0], 1, 1))
