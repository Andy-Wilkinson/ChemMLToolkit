from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from chemmltoolkit.utils.list_utils import zip_expand
from chemmltoolkit.tensorflow.utils import register_keras_custom_object
from chemmltoolkit.tensorflow.blocks.utils import get_batchnorm
from chemmltoolkit.tensorflow.blocks.utils import get_dropout


@register_keras_custom_object
class DenseBlock(Layer):
    """A block of Dense layers

    A stacked block of Dense layers, with optional batch normalisation and
    dropout.

    If you supply a list to any argument, this specifies the value to use for
    each individual layer. If you supply a single value, this same value
    applies to all layers.

    Arguments:
        units: Positive integer, dimensionality of the hidden layer.
        activation: Activation function to use.
        batchnorm: The type of batch normalisation to use. Possible values are,
            None, 'norm' (normalisation), 'renorm' (renormalisation).
        dropout: The level of dropout to use (the default of 0.0 does not
            apply any dropout).
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., last_units)`.
    """
    def __init__(self,
                 units,
                 activation=None,
                 batchnorm=None,
                 dropout=0.0,
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        self.dense_layers = []
        self.batchnorm_layers = []
        self.dropout_layers = []

        for units, activation, batchnorm, dropout \
                in zip_expand(units, activation, batchnorm, dropout):

            dense_layer = Dense(units,
                                activation=activation,
                                use_bias=not batchnorm)

            self.dense_layers.append(dense_layer)
            self.batchnorm_layers.append(get_batchnorm(batchnorm))
            self.dropout_layers.append(get_dropout(dropout))

    def call(self, inputs):
        x = inputs

        for dense, batchnorm, dropout \
                in zip(self.dense_layers,
                       self.batchnorm_layers,
                       self.dropout_layers):

            x = dense(x)
            x = batchnorm(x) if batchnorm else x
            x = dropout(x) if dropout else x

        return x
