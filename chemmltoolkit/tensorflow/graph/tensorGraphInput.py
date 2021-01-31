from tensorflow.keras import Input
from chemmltoolkit.tensorflow.graph import TensorGraphShape
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES


def TensorGraphInput(shape: TensorGraphShape,
                     batch_size=None,
                     name: str = 'graph'):
    """Returns a Keras input for TensorGraphs.

    This function simply returns a dictionary with keras Input elements
    for each tensor in the input TensorGraph.

    Arguments:
        shape: The shape of the input graph.
        batch_size: Optional static batch size (integer).
        name: An optional prefix for the names of the Input layers. The
            default value is 'graph'.
    """
    return {
        NODE_FEATURES: Input(
            shape=(shape.num_nodes, shape.node_dims),
            batch_size=batch_size,
            name=f'{name}_node_features'),
        EDGE_FEATURES: Input(
            shape=(shape.edge_dims, shape.num_nodes, shape.num_nodes),
            batch_size=batch_size,
            name=f'{name}_edge_features')
    }
