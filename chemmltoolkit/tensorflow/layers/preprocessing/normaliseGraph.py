import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer
from chemmltoolkit.tensorflow.graph.tensorGraph import map_edge_features
from chemmltoolkit.tensorflow.utils import register_keras_custom_object


@register_keras_custom_object
class NormaliseGraph(PreprocessingLayer):
    """A preprocessing layer to normalise TensorGraphs.
    Arguments:
        method: The method of normalisation to use. Possible options are,
            'normal' (default): Scales all rows to sum to one
            'spectral': Uses the spectral method of Kipf and Welling
                (https://arxiv.org/abs/1609.02907)
    Input shape:
        A TensorGraph is provided as input.
    Output shape:
        A TensorGraph with identical shape to the input.
    """

    def __init__(self, name=None, method='normal', **kwargs):
        super(NormaliseGraph, self).__init__(name=name, **kwargs)
        self.method = method

    def call(self, inputs):
        if self.method == 'spectral':
            map_fn = self.normalise_spectral
        else:
            map_fn = self.normalise_normal

        return map_edge_features(inputs, map_fn)

    def normalise_normal(self, edge_features):
        degree = tf.reduce_sum(edge_features, axis=-2)
        degree = tf.expand_dims(degree, -1)
        return tf.math.divide_no_nan(edge_features, degree)

    def normalise_spectral(self, edge_features):
        degree = tf.reduce_sum(edge_features, axis=-2)
        degree_inv = tf.pow(degree, -0.5)
        degree_inv = tf.where(tf.math.is_inf(degree_inv), 0.0, degree_inv)

        # (degree_inv)^T . matrix . (degree_inv)
        di1 = tf.expand_dims(degree_inv, -2)
        di2 = tf.expand_dims(degree_inv, -1)
        return tf.multiply(tf.multiply(di1, edge_features), di2)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(NormaliseGraph, self).get_config()
        config.update({'method': self.method})
        return config
