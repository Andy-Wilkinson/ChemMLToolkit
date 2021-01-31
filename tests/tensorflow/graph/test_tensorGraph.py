import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraph
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES
from tests.test_utils.math_utils import generate_random_node_features
from tests.test_utils.math_utils import generate_random_adjacency


class TestTensorGraph(tf.test.TestCase):
    def test_node_and_edge_features(self):
        random = np.random.RandomState(seed=42)
        node_features = generate_random_node_features(random, 3, 10, 42)
        edge_features = generate_random_adjacency(random, 3, 10, 7)

        node_tensor = tf.convert_to_tensor(node_features, dtype=tf.float32)
        edge_tensor = tf.convert_to_tensor(edge_features, dtype=tf.float32)

        graph = TensorGraph(node_tensor, edge_tensor)

        self.assertAllEqual(node_tensor, graph[NODE_FEATURES])
        self.assertAllEqual(edge_tensor, graph[EDGE_FEATURES])
