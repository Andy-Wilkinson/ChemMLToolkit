from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LayerNormalization
from chemmltoolkit.utils.list_utils import zip_expand
from chemmltoolkit.tensorflow.utils import register_keras_custom_object
from chemmltoolkit.tensorflow.blocks.utils import get_dropout


@register_keras_custom_object
class DilConv1DWithGLUBlock(Layer):
    """A block of dilational Conv1D layers with GLUs.

    A stacked block of dilational layers with gated linear units (GLUs) as
    described in Nature Biotech., 2019, 37, 1038.

    If you supply a list to any argument, this specifies the value to use for
    each individual layer. If you supply a single value, this same value
    applies to all layers.

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        dilation_rate: Integer, the dilation for the convolution. The default
            value of None will use a series of dilations starting from one and
            doubling for each consecutive layer.
        activation: Activation function to use.
        residual_connection: Boolean, whether to include a residual connection.
        dropout: The level of dropout to use (the default of 0.0 does not
            apply any dropout).
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`.
    Output shape:
        3D tensor with shape: `(batch_size, steps, last_filter)`.
    """

    def __init__(self,
                 filters,
                 kernel_size=2,
                 dilation_rate=None,
                 activation=None,
                 residual_connection=True,
                 dropout=0.0,
                 **kwargs):
        super(DilConv1DWithGLUBlock, self).__init__(**kwargs)

        if dilation_rate is None:
            dilation_rate = [2**n for n in range(len(filters))]

        self.conv_layers = []
        self.dropout_layers = []

        for filters, kernel_size, dilation_rate, activation, \
                residual, dropout \
                in zip_expand(filters, kernel_size, dilation_rate, activation,
                              residual_connection, dropout):

            conv_layer = DilConv1DWithGLU(filters,
                                          kernel_size=kernel_size,
                                          dilation_rate=dilation_rate,
                                          activation=activation,
                                          residual_connection=residual)
            self.conv_layers.append(conv_layer)
            self.dropout_layers.append(get_dropout(dropout))

    def call(self, inputs):
        x = inputs

        for conv, dropout in zip(self.conv_layers, self.dropout_layers):
            x = conv(x)
            x = dropout(x) if dropout else x

        return x


class DilConv1DWithGLU(Layer):
    """A dilational Conv1D layer with GLU.

    This layer consists of,
      - A 1x1 Conv1D layer
      - A dilational Conv1D layer
      - A gated linear unit (GLU)
      - An optional residual connection

    Typically these blocks will be stacked with the dilation rate doubling
    for each consecutive layer.

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window (default=2).
        dilation_rate: Integer, the dilation for the convolution.
        activation: Activation function to use.
        residual_connection: Boolean, whether to include a residual connection.

    """

    def __init__(self, filters, kernel_size=2, dilation_rate=1,
                 activation=None, residual_connection=True, **kwargs):

        super(DilConv1DWithGLU, self).__init__(**kwargs)

        self.start_ln = LayerNormalization()
        self.start_conv1x1 = Conv1D(filters, 1)

        self.dilconv_ln = LayerNormalization()
        self.dilated_conv = Conv1D(filters,
                                   kernel_size,
                                   dilation_rate=dilation_rate,
                                   padding='causal')

        self.gate_ln = LayerNormalization()
        self.end_conv1x1 = Conv1D(filters, 1)
        self.gated_conv1x1 = Conv1D(filters, 1, activation='sigmoid')

        self.activation = activation
        self.residual_connection = residual_connection

    def call(self, inputs):
        # Input shape [batch_size, num_channels, max_seq_len]

        # Applying 1x1 convolution
        x = self.start_ln(inputs)
        x = self.activation(x) if self.activation else x
        x = self.start_conv1x1(x)

        # Applying dilated convolution
        x = self.dilconv_ln(x)
        x = self.activation(x) if self.activation else x
        x = self.dilated_conv(x)

        # Applying gated linear unit
        x = self.gate_ln(x)
        x = self.activation(x) if self.activation else x
        x = self.end_conv1x1(x) * self.gated_conv1x1(x)

        # If residual connection
        if self.residual_connection:
            x = x + inputs

        return x
