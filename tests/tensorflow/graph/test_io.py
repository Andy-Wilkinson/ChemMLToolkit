import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraphShape
from chemmltoolkit.tensorflow.graph.io import define_tfrecord_features
from chemmltoolkit.tensorflow.graph.io import tfrecord_to_tensorgraph
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES
from tests.test_utils.math_utils import generate_random_node_features
from tests.test_utils.math_utils import generate_random_adjacency


class TestIo(tf.test.TestCase):
    def test_define_tfrecord_features(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        feat = define_tfrecord_features(shape)

        assert feat == {
            'graph:num_nodes': tf.io.FixedLenFeature([], tf.int64),
            'graph:node_features': tf.io.VarLenFeature(tf.float64),
            'graph:edge_features': tf.io.VarLenFeature(tf.float32),
        }

    def test_define_tfrecord_features_with_name(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        feat = define_tfrecord_features(shape, name='test')

        assert feat == {
            'test:num_nodes': tf.io.FixedLenFeature([], tf.int64),
            'test:node_features': tf.io.VarLenFeature(tf.float64),
            'test:edge_features': tf.io.VarLenFeature(tf.float32),
        }

    def test_tfrecord_to_tensorgraph(self):
        random = np.random.RandomState(seed=42)
        num_nodes = 10
        node_features = generate_random_node_features(
            random, None, num_nodes, 42)
        edge_features = generate_random_adjacency(random, None, num_nodes, 7)

        example = {
            'graph:num_nodes': tf.convert_to_tensor(num_nodes),
            'graph:node_features':
                tf.sparse.from_dense(node_features.flatten()),
            'graph:edge_features':
                tf.sparse.from_dense(edge_features.flatten())
        }

        graph = tfrecord_to_tensorgraph(example)

        self.assertAllEqual(node_features, graph[NODE_FEATURES])
        self.assertAllEqual(edge_features, graph[EDGE_FEATURES])

    def test_tfrecord_to_tensorgraph_with_name(self):
        random = np.random.RandomState(seed=42)
        num_nodes = 10
        node_features = generate_random_node_features(
            random, None, num_nodes, 42)
        edge_features = generate_random_adjacency(random, None, num_nodes, 7)

        example = {
            'test:num_nodes': tf.convert_to_tensor(num_nodes),
            'test:node_features':
                tf.sparse.from_dense(node_features.flatten()),
            'test:edge_features':
                tf.sparse.from_dense(edge_features.flatten())
        }

        graph = tfrecord_to_tensorgraph(example, name='test')

        self.assertAllEqual(node_features, graph[NODE_FEATURES])
        self.assertAllEqual(edge_features, graph[EDGE_FEATURES])
