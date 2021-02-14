from chemmltoolkit.tensorflow.graph.tensorGraph import TensorGraphShape
import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraph
from chemmltoolkit.tensorflow.graph import TensorGraphInput
from chemmltoolkit.tensorflow.layers.preprocessing import AddSelfLoops
from tests.test_utils.math_utils import generate_random_node_features
from tests.test_utils.tensorgraph_utils import assert_graph_equal
from tests.test_utils.tensorgraph_utils import assert_graph_shape
from tests.test_utils.tensorgraph_utils import graph_tensor_shape


class TestAddSelfLoops(tf.test.TestCase):
    def test_call_without_batches(self):
        input = np.array([
            [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 2, 3], [2, 0, 4], [3, 4, 0]],
        ], dtype=np.float)
        expected_output = np.array([
            [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
            [[1, 2, 3], [2, 1, 4], [3, 4, 1]],
        ], dtype=np.float)

        self._test_call(input, expected_output)

    def test_call_with_batches(self):
        input = np.array([
            [
                [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 2, 3], [2, 0, 4], [3, 4, 0]],
            ],
            [
                [[0, 3, 4], [1, 0, 1], [0, 1, 0]],
                [[0, 1, 3], [2, 0, 4], [3, 4, 0]],
            ]
        ], dtype=np.float)
        expected_output = np.array([
            [
                [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
                [[1, 2, 3], [2, 1, 4], [3, 4, 1]],
            ],
            [
                [[1, 3, 4], [1, 1, 1], [0, 1, 1]],
                [[1, 1, 3], [2, 1, 4], [3, 4, 1]],
            ]
        ], dtype=np.float)

        self._test_call(input, expected_output)

    def _test_call(self, input, expected_output, **kwargs):
        graph_input = TensorGraph(None, input)
        graph_expected = TensorGraph(None, expected_output)

        shiftTensor = AddSelfLoops(**kwargs)
        input_shape = graph_tensor_shape(graph_input)
        computed_shape = shiftTensor.compute_output_shape(input_shape)
        graph_output = shiftTensor(graph_input)

        assert_graph_equal(self, graph_expected, graph_output)
        assert_graph_shape(self, graph_output, computed_shape)

    def test_serialize(self):
        random = np.random.RandomState(seed=42)
        node_features = generate_random_node_features(random, 1, 3, 1)

        input = TensorGraph(node_features, np.array([[
            [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 2, 3], [2, 0, 4], [3, 4, 0]],
        ]], dtype=np.float))
        expected_output = TensorGraph(node_features, np.array([[
            [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
            [[1, 2, 3], [2, 1, 4], [3, 4, 1]],
        ]], dtype=np.float))

        model_input = TensorGraphInput(
            TensorGraphShape(num_nodes=3, edge_dims=2))
        x = AddSelfLoops()(model_input)
        model = tf.keras.Model(model_input, x)
        config = model.get_config()

        reinitialized_model = tf.keras.Model.from_config(config)
        output = reinitialized_model(input)

        assert_graph_equal(self, expected_output, output,
                           allow_nodes_close=True)
