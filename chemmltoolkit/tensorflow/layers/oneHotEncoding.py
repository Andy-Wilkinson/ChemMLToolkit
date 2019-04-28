import tensorflow as tf
from chemmltoolkit.tensorflow.utils import register_keras_custom_object


@register_keras_custom_object
class OneHotEncoding(tf.keras.layers.Layer):
    """Applys one-hot encoding to the input.
    Arguments:
        depth: A scalar defining the depth of the one hot dimension.
        on_value: A scalar defining the value to fill in output when
            indices[j] = i. (default: 1)
        off_value: A scalar defining the value to fill in output when
            indices[j] != i. (default: 0)
        axis: The axis to fill (default: -1, a new inner-most axis).
        output_dtype: The data type of the output tensor.
    Input shape:
        nD tensor with shape: `(batch_size, ...)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., depth)`.
    """

    def __init__(self,
                 depth,
                 on_value=None,
                 off_value=None,
                 axis=None,
                 output_dtype=None,
                 **kwargs):
        super(OneHotEncoding, self).__init__(**kwargs)
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.output_dtype = output_dtype

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        outputs = tf.one_hot(inputs, self.depth, self.on_value, self.off_value,
                             self.axis, self.output_dtype)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.axis:
            output_shape = input_shape[:self.axis] + \
                [self.depth] + \
                input_shape[self.axis:]
        else:
            output_shape = input_shape + [self.depth]
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super(OneHotEncoding, self).get_config()
        config.update({'depth': self.depth,
                       'on_value': self.on_value,
                       'off_value': self.off_value,
                       'axis': self.axis,
                       'output_dtype': self.output_dtype})
        return config
