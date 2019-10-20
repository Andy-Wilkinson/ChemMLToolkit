from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Multiply
from chemmltoolkit.utils.list_utils import zip_expand
from chemmltoolkit.tensorflow.utils import register_keras_custom_object
from chemmltoolkit.tensorflow.blocks.utils import get_batchnorm
from chemmltoolkit.tensorflow.blocks.utils import get_dropout


@register_keras_custom_object
class RecurrentBlock(Layer):
    """A block of recurrent layers

    A stacked block of GRU or LSTM layers, with optional batch normalisation
    and dropout.

    If you supply a list to any argument, this specifies the value to use for
    each individual layer. If you supply a single value, this same value
    applies to all layers.

    Arguments:
        cell_type: The type of recurrent cell to use ('gru' or 'lstm')
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        bidirectional: Boolean, whether the layer should be bidirectional.
        conditioning_mode: The way in which the conditioning tensor is
            combined with the current input for each layer. Possible values
            are, None (do not apply conditioning), 'ave', 'concat', 'mul',
            'sum'.
        batchnorm: The type of batch normalisation to use. Possible values are,
            None, 'norm' (normalisation), 'renorm' (renormalisation).
        dropout: The level of dropout to use (the default of 0.0 does not
            apply any dropout).
    Input shape:
        Two tensors are provided as input.
            3D input tensor with shape:
                `(batch_size, steps, input_dim)`.
            3D conditioning tensor with shape:
                `(batch_size, steps, input_dim)`
                or None if conditioning is not used.
    Output shape:
        3D tensor with shape: `(batch_size, steps, last_units)`
    """
    def __init__(self,
                 cell_type,
                 units,
                 activation='tanh',
                 bidirectional=False,
                 conditioning_mode=None,
                 batchnorm=None,
                 dropout=0.0,
                 **kwargs):
        super(RecurrentBlock, self).__init__(**kwargs)

        self.conditioning_layers = []
        self.recurrent_layers = []
        self.batchnorm_layers = []
        self.dropout_layers = []

        for cell_type, units, activation, bidirectional, conditioning_mode, \
                batchnorm, dropout \
                in zip_expand(cell_type, units, activation, bidirectional,
                              conditioning_mode, batchnorm, dropout):

            if conditioning_mode is None:
                conditioning_layer = None
            elif conditioning_mode == 'ave':
                conditioning_layer = Average()
            elif conditioning_mode == 'concat':
                conditioning_layer = Concatenate()
            elif conditioning_mode == 'mul':
                conditioning_layer = Multiply()
            elif conditioning_mode == 'sum':
                conditioning_layer = Add()
            else:
                raise ValueError(f'Invalid value `{conditioning_mode}` for' +
                                 '`conditioning_mode` parameter')

            if cell_type == 'gru':
                recurrent_layer = GRU(units,
                                      activation=activation,
                                      use_bias=not batchnorm,
                                      return_sequences=True)
            elif cell_type == 'lstm':
                recurrent_layer = LSTM(units,
                                       activation=activation,
                                       use_bias=not batchnorm,
                                       return_sequences=True)
            else:
                raise ValueError(f'Invalid value `{cell_type}` for' +
                                 '`cell_type` parameter')

            if bidirectional:
                recurrent_layer = Bidirectional(recurrent_layer)

            self.conditioning_layers.append(conditioning_layer)
            self.recurrent_layers.append(recurrent_layer)
            self.batchnorm_layers.append(get_batchnorm(batchnorm))
            self.dropout_layers.append(get_dropout(dropout))

    def call(self, inputs):
        x, conditioning_sequence = inputs

        for conditioning, layer, batchnorm, dropout \
                in zip(self.conditioning_layers,
                       self.recurrent_layers,
                       self.batchnorm_layers,
                       self.dropout_layers):

            x = conditioning([x, conditioning_sequence]) if conditioning else x
            x = layer(x)
            x = batchnorm(x) if batchnorm else x
            x = dropout(x) if dropout else x

        return x
