import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraphShape
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES


class TestTensorGraphShape(tf.test.TestCase):
    def test_constructor_defaults(self):
        shape = TensorGraphShape()

        assert shape.num_nodes is None
        assert shape.node_dims == 1
        assert shape.edge_dims == 1
        assert shape.node_dtype == tf.float32
        assert shape.edge_dtype == tf.float32

    def test_constructor_with_values(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.int32)

        assert shape.num_nodes == 42
        assert shape.node_dims == 8
        assert shape.edge_dims == 6
        assert shape.node_dtype == tf.float64
        assert shape.edge_dtype == tf.int32

    def test_get_padding_shapes(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6)

        graph_padding = shape.get_padding_shapes()
        padding_node = graph_padding[NODE_FEATURES]
        padding_edge = graph_padding[EDGE_FEATURES]

        self.assertAllEqual(padding_node, (42, 8))
        self.assertAllEqual(padding_edge, (6, 42, 42))

    def test_get_padding_values(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6)

        graph_padding = shape.get_padding_values()
        padding_node = graph_padding[NODE_FEATURES]
        padding_edge = graph_padding[EDGE_FEATURES]

        self.assertAllEqual(padding_node, 0.0)
        self.assertAllEqual(padding_edge, 0.0)
