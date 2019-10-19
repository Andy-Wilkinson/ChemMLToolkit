from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from chemmltoolkit.utils.list_utils import zip_expand
from chemmltoolkit.tensorflow.utils import register_keras_custom_object
from chemmltoolkit.tensorflow.blocks.utils import get_batchnorm
from chemmltoolkit.tensorflow.blocks.utils import get_dropout


@register_keras_custom_object
class Conv1DBlock(Layer):
    """A block of Conv1D layers

    A stacked block of Conv1D layers, with optional batch normalisation and
    dropout.

    If you supply a list to any argument, this specifies the value to use for
    each individual layer. If you supply a single value, this same value
    applies to all layers.

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        activation: Activation function to use.
        batchnorm: The type of batch normalisation to use. Possible values are,
            None, 'norm' (normalisation), 'renorm' (renormalisation).
        dropout: The level of dropout to use (the default of 0.0 does not
            apply any dropout).
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`.
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, last_filter)`
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 activation=None,
                 batchnorm=None,
                 dropout=0.0,
                 **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)

        self.conv_layers = []
        self.batchnorm_layers = []
        self.dropout_layers = []

        for filters, kernel_size, activation, batchnorm, dropout \
                in zip_expand(filters, kernel_size, activation,
                              batchnorm, dropout):

            conv_layer = Conv1D(filters,
                                kernel_size,
                                activation=activation,
                                use_bias=not batchnorm)
            self.conv_layers.append(conv_layer)
            self.batchnorm_layers.append(get_batchnorm(batchnorm))
            self.dropout_layers.append(get_dropout(dropout))

    def call(self, inputs):
        x = inputs

        for conv, batchnorm, dropout \
                in zip(self.conv_layers,
                       self.batchnorm_layers,
                       self.dropout_layers):

            x = conv(x)
            x = batchnorm(x) if batchnorm else x
            x = dropout(x) if dropout else x

        return x
