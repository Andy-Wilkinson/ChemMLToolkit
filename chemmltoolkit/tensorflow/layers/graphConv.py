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
from chemmltoolkit.tensorflow.graph.tensorGraph import NODE_FEATURES
from chemmltoolkit.tensorflow.graph.tensorGraph import EDGE_FEATURES
from chemmltoolkit.tensorflow.utils import register_keras_custom_object


@register_keras_custom_object
class GraphConv(Layer):
    """A graph convolutional layer.
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, RELU activation is applied.
        use_bias: Boolean, whether the layer uses a bias vector.
        add_self_loops: Boolean, whether to add a separate weight matrix
            for self loops.
        aggregation_method: Specifies the method to use when combining outputs
            if multiple adjacency matricies are specified ('concat', 'sum' or
            'max').
        graph_regularization: The type of graph regularization to use. The
            default (None) is to not use any regularization. Possible values
            are 'basis' (basis decomposition).
        num_bases: The number of bases to use for graph regularization. This
            must be set if graph_regularization is used, otherwise left as the
            default (None).
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_coef_initializer: Initializer for the `kernel` coefficients
            matrix (only used with graph regularization).
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_coef_regularizer: Regularizer function applied to
            the `kernel` coefficients matrix (only used with graph
            regularization).
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to
            the `kernel` coefficients matrix (only used with graph
            regularization).
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        Two tensors are provided as input.
        An N-D feature tensor with shape:
            `(batch_size, ..., input_dim)`.
        A 3-D adjacency tensor with shape:
            `(batch_size, num_adjacency, input_dim, input_dim)`.
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 add_self_loops=False,
                 aggregation_method='sum',
                 graph_regularization=None,
                 num_bases=None,
                 kernel_initializer='glorot_uniform',
                 kernel_coef_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 kernel_coef_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 kernel_coef_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConv, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        if graph_regularization not in [None, 'basis']:
            raise ValueError('An invalid value of `graph_regularization` has' +
                             ' been passed to a `GraphConv` layer.')

        if graph_regularization is not None and num_bases is None:
            raise ValueError('The `num_bases` property must be set if ' +
                             '`graph_regularization` is not None.')

        if graph_regularization is None and num_bases is not None:
            raise ValueError('The `num_bases` property must not be set if ' +
                             '`graph_regularization` is None.')

        if num_bases is not None and num_bases <= 0:
            raise ValueError('The `num_bases` property must be a positive ' +
                             'integer.')

        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.add_self_loops = add_self_loops
        self.aggregation_method = aggregation_method
        self.graph_regularization = graph_regularization
        self.num_bases = num_bases
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_coef_initializer = initializers.get(
            kernel_coef_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_coef_regularizer = regularizers.get(
            kernel_coef_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_coef_constraint = constraints.get(kernel_coef_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = {
            NODE_FEATURES: InputSpec(min_ndim=2),
            EDGE_FEATURES: InputSpec(ndim=4)
        }

    def build(self, input_shape):
        # Validate inputs
        if not isinstance(input_shape, dict):
            raise ValueError('The `GraphConv` layer expects a dict of inputs.')
        if NODE_FEATURES not in input_shape:
            raise ValueError('A `GraphConv` layer should be called with a '
                             'dict containing key "' + NODE_FEATURES + '".')
        if EDGE_FEATURES not in input_shape:
            raise ValueError('A `GraphConv` layer should be called with a '
                             'dict containing key "' + EDGE_FEATURES + '".')

        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `GraphConv` layer with '
                            'non-floating point dtype %s' % (dtype,))

        features_shape = tensor_shape.TensorShape(input_shape[NODE_FEATURES])
        adjacency_shape = tensor_shape.TensorShape(input_shape[EDGE_FEATURES])
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

        if self.num_bases \
                and self.num_bases > \
                tensor_shape.dimension_value(adjacency_shape[1]):
            raise ValueError('The adjacency matrix (second input) passed to ' +
                             '`GraphConv` must not have less adjacency ' +
                             'matricies than `num_bases`.')

        # Define the input spec
        self.num_node_features = tensor_shape.dimension_value(
            features_shape[-1])
        self.num_nodes = tensor_shape.dimension_value(adjacency_shape[2])
        self.num_edge_features = tensor_shape.dimension_value(
            adjacency_shape[1])

        self.input_spec = {
            NODE_FEATURES: InputSpec(min_ndim=2, axes={
                -1: self.num_node_features,
                -2: self.num_nodes}),
            EDGE_FEATURES: InputSpec(ndim=4, axes={
                1: self.num_edge_features,
                2: self.num_nodes,
                3: self.num_nodes})
        }

        # Add weights
        num_self_loops = 1 if self.add_self_loops else 0
        self.num_weights = self.num_edge_features + num_self_loops
        num_kernels = self.num_bases if self.graph_regularization else \
            self.num_weights

        self.kernel = self.add_weight(
            'kernel',
            shape=[num_kernels,
                   self.num_node_features,
                   self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        if self.graph_regularization:
            self.kernel_coef = self.add_weight(
                'kernel_coef',
                shape=[self.num_weights,
                       self.num_bases],
                initializer=self.kernel_coef_initializer,
                regularizer=self.kernel_coef_regularizer,
                constraint=self.kernel_coef_constraint,
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
        features = tf.convert_to_tensor(inputs[NODE_FEATURES])
        is_sparse_adjacency = isinstance(inputs[EDGE_FEATURES],
                                         tf.sparse.SparseTensor)
        if is_sparse_adjacency:
            adjacency = tf.sparse.to_dense(inputs[EDGE_FEATURES])
        else:
            adjacency = tf.convert_to_tensor(inputs[EDGE_FEATURES])

        weights = self._get_weight_matrix()

        features = tf.expand_dims(features, 1)
        outputs = tf.matmul(adjacency, features)

        if self.add_self_loops:
            outputs = tf.concat([features, outputs], 1)

        outputs = tf.matmul(outputs, weights)

        if self.aggregation_method == 'concat':
            outputs = tf.transpose(outputs, (0, 2, 1, 3))
            outputs_shape = tf.shape(outputs)
            outputs = tf.reshape(outputs,
                                 (outputs_shape[0], outputs_shape[1], -1))
        elif self.aggregation_method == 'sum':
            outputs = tf.reduce_sum(outputs, axis=1)
        elif self.aggregation_method == 'max':
            outputs = tf.reduce_max(outputs, axis=1)
        else:
            raise ValueError('Undefined aggregation method' +
                             f'`{self.aggregation_method}`.')

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return {**inputs, NODE_FEATURES: outputs}

    def _get_weight_matrix(self):
        # Get the weight matrix by applying the specified regularization
        # method (if any) to the kernel matrix
        if self.graph_regularization == 'basis':
            weights = tf.reshape(self.kernel, (self.num_bases,
                                               self.num_node_features *
                                               self.units))
            weights = tf.matmul(self.kernel_coef, weights)
            weights = tf.reshape(weights, (self.num_weights,
                                           self.num_node_features, self.units))
            return weights
        else:
            return self.kernel

    def compute_output_shape(self, input_shape):
        features_shape = tensor_shape.TensorShape(input_shape[NODE_FEATURES])
        adjacency_shape = tensor_shape.TensorShape(input_shape[EDGE_FEATURES])
        features_shape = features_shape.with_rank_at_least(2)

        if self.aggregation_method == 'concat':
            multiplier = tensor_shape.dimension_value(adjacency_shape[1])
        else:
            multiplier = 1

        out_shape = features_shape[:-1].concatenate(self.units * multiplier)

        return {
            NODE_FEATURES: out_shape,
            EDGE_FEATURES: adjacency_shape
        }

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'add_self_loops': self.add_self_loops,
            'aggregation_method': self.aggregation_method,
            'graph_regularization': self.graph_regularization,
            'num_bases': self.num_bases,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'kernel_coef_initializer':
                initializers.serialize(self.kernel_coef_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'kernel_coef_regularizer':
                regularizers.serialize(self.kernel_coef_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'kernel_coef_constraint':
                constraints.serialize(self.kernel_coef_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
