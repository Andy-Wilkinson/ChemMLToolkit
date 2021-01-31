import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraphInput
from chemmltoolkit.tensorflow.graph import TensorGraphShape
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES


class TestTensorGraphInput(tf.test.TestCase):
    def test_simple_graph(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        input = TensorGraphInput(shape)
        input_node_features = input[NODE_FEATURES]
        input_edge_features = input[EDGE_FEATURES]

        self.assertAllEqual(input_node_features.shape, (None, 42, 8))
        self.assertAllEqual(input_edge_features.shape, (None, 6, 42, 42))

        assert input_node_features.name.startswith('graph_node_features')
        assert input_edge_features.name.startswith('graph_edge_features')

    def test_with_batch_size(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        input = TensorGraphInput(shape, batch_size=21)
        input_node_features = input[NODE_FEATURES]
        input_edge_features = input[EDGE_FEATURES]

        self.assertAllEqual(input_node_features.shape, (21, 42, 8))
        self.assertAllEqual(input_edge_features.shape, (21, 6, 42, 42))

        assert input_node_features.name.startswith('graph_node_features')
        assert input_edge_features.name.startswith('graph_edge_features')

    def test_with_name(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        input = TensorGraphInput(shape, name='example')
        input_node_features = input[NODE_FEATURES]
        input_edge_features = input[EDGE_FEATURES]

        self.assertAllEqual(input_node_features.shape, (None, 42, 8))
        self.assertAllEqual(input_edge_features.shape, (None, 6, 42, 42))

        assert input_node_features.name.startswith('example_node_features')
        assert input_edge_features.name.startswith('example_edge_features')
