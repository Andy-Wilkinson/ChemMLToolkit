import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES
from chemmltoolkit.tensorflow.utils import register_keras_custom_object


@register_keras_custom_object
class AddSelfLoops(PreprocessingLayer):
    """A preprocessing layer to add self-loops to TensorGraphs.

    Self-loops are additional connections between each node and itself where
    the edge features are all one.

    Input shape:
        A TensorGraph is provided as input.
    Output shape:
        A TensorGraph with identical shape to the input.
    """

    def __init__(self, name=None, **kwargs):
        super(AddSelfLoops, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        edge_features = inputs[EDGE_FEATURES]

        eye = tf.eye(tf.shape(edge_features)[-1])
        edge_features = edge_features + eye

        return {**inputs, EDGE_FEATURES: edge_features}

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(AddSelfLoops, self).get_config()
        return config
