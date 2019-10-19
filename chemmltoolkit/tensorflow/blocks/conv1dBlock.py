from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from chemmltoolkit.utils.list_utils import zip_expand
from chemmltoolkit.tensorflow.utils import register_keras_custom_object
from chemmltoolkit.tensorflow.blocks.utils import get_batchnorm
from chemmltoolkit.tensorflow.blocks.utils import get_dropout


@register_keras_custom_object
class Conv1DBlock(Layer):
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
