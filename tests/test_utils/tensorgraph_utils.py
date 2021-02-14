import tensorflow as tf
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES


def assert_graph_equal(test_case: tf.test.TestCase, a: dict, b: dict,
                       allow_nodes_close=False, allow_edges_close=False):
    keys = set(a.keys()) | set(b.keys())

    for key in keys:
        if (key == NODE_FEATURES and allow_nodes_close) or \
                (key == EDGE_FEATURES and allow_edges_close):
            test_case.assertAllClose(a[key], b[key], 0.02)
        else:
            test_case.assertAllEqual(a[key], b[key])


def assert_graph_shape(test_case: tf.test.TestCase, graph: dict, shape):
    graph_shape = graph_tensor_shape(graph)
    test_case.assertAllEqual(shape, graph_shape)


def graph_tensor_shape(graph: dict):
    def _get_shape(tensor):
        if tensor is None:
            return None
        return tensor.shape

    return {k: _get_shape(t) for k, t in graph.items()}
