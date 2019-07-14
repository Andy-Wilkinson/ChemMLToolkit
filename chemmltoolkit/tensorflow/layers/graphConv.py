import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.backend as K
from chemmltoolkit.tensorflow.utils import register_keras_custom_object


@register_keras_custom_object
class GraphConv(Layer):
    """A graph convolutional layer.
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, RELU activation is applied.
        use_bias: Boolean, whether the layer uses a bias vector.
        add_adjacency_self_loops: Boolean, whether to add self loops
            to the adjacency matrix.
        normalize_adjacency_matrix: Boolean, whether to normalize the
            adjacency matrix.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        Two tensors are provided as input.
        An N-D feature tensor with shape:
            `(batch_size, ..., input_dim)`.
        A 2-D adjacency tensor with shape:
            `(batch_size, ..., input_dim, input_dim)`.
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """
    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 add_adjacency_self_loops=False,
                 normalize_adjacency_matrix=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConv, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.add_adjacency_self_loops = add_adjacency_self_loops
        self.normalize_adjacency_matrix = normalize_adjacency_matrix
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = [InputSpec(min_ndim=2), InputSpec(ndim=3)]

    def build(self, input_shape):
        # Validate inputs
        if not isinstance(input_shape, list):
            raise ValueError('The `GraphConv` layer expects a list of inputs.')
        if len(input_shape) != 2:
            raise ValueError('A `GraphConv` layer should be called with 2'
                             'inputs. Got ' + str(len(input_shape)) +
                             ' inputs.')

        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `GraphConv` layer with '
                            'non-floating point dtype %s' % (dtype,))

        features_shape = tensor_shape.TensorShape(input_shape[0])
        adjacency_shape = tensor_shape.TensorShape(input_shape[1])
        if tensor_shape.dimension_value(features_shape[-1]) is None:
            raise ValueError('The last dimension of the features matrix '
                             '(first input) to `GraphConv` should be defined.'
                             'Found `None`.')
        if tensor_shape.dimension_value(features_shape[-2]) != \
                tensor_shape.dimension_value(adjacency_shape[-1]) \
                or tensor_shape.dimension_value(features_shape[-2]) != \
                tensor_shape.dimension_value(adjacency_shape[-2]):
            raise ValueError('The last two dimensions of the adjacency matrix'
                             '(second input) to `GraphConv` should be the '
                             'same as the penultimate dimension of the '
                             'features matrix (first input). '
                             f'Found {adjacency_shape} for feature size ' +
                             tensor_shape.dimension_value(features_shape[-2]))

        # Define the input spec
        last_dim = tensor_shape.dimension_value(features_shape[-1])
        adjacency_dim = tensor_shape.dimension_value(features_shape[-2])
        self.input_spec = [
            InputSpec(min_ndim=2, axes={-1: last_dim, -2: adjacency_dim}),
            InputSpec(ndim=3, axes={-1: adjacency_dim, -2: adjacency_dim})
        ]

        # Add weights
        self.kernel = self.add_weight(
                    'kernel',
                    shape=[last_dim, self.units],
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    dtype=self.dtype,
                    trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                        'bias',
                        shape=[self.units, ],
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        dtype=self.dtype,
                        trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        features = tf.convert_to_tensor(inputs[0])
        adjacency = tf.convert_to_tensor(inputs[1])

        # Add self-loops and normalize the adjacency matrices if required
        adjacency = self.preprocess_adjacency_matrices(
            adjacency,
            add_adjacency_self_loops=self.add_adjacency_self_loops,
            normalize_adjacency_matrix=self.normalize_adjacency_matrix)

        # Apply the adjacency matrix
        # NB: Do 'tensordot' rather than 'matmul' as W is not
        # broadcasted across batches
        outputs = tf.matmul(adjacency, features)
        outputs = tf.tensordot(outputs, self.kernel, axes=1)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return outputs

    def compute_output_shape(self, input_shape):
        features_shape = tensor_shape.TensorShape(input_shape[0])
        features_shape = features_shape.with_rank_at_least(2)

        return features_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'add_adjacency_self_loops': self.add_adjacency_self_loops,
            'normalize_adjacency_matrix': self.normalize_adjacency_matrix,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def preprocess_adjacency_matrices(adjacency,
                                      add_adjacency_self_loops=False,
                                      normalize_adjacency_matrix=False):
        """Preprocesses the adjacency matrices.

        Note you should set the corresponding flags to False in the
        GraphConv layer initialiser if they have already been
        preprocessed.

        Args:
            adjacency: The adjacency matrices to process.
            add_adjacency_self_loops: Boolean, whether to add self loops
                to the adjacency matrix.
            normalize_adjacency_matrix: Boolean, whether to normalize the
                adjacency matrix.

        Returns:
            The processed adjacency matrices.
        """
        if add_adjacency_self_loops:
            adjacency = tf.linalg.set_diag(adjacency,
                                           tf.ones_like(adjacency)[:, 0])
        if normalize_adjacency_matrix:
            degree_inverse = tf.linalg.diag(1/tf.reduce_sum(adjacency, axis=1))
            adjacency = tf.matmul(degree_inverse, adjacency)

        return adjacency
