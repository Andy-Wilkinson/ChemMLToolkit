import tensorflow as tf
from chemmltoolkit.tensorflow.utils import register_keras_custom_object


@register_keras_custom_object
class ShiftTensor(tf.keras.layers.Layer):
    """Shifts the input tensor along the time axis.
    This layer moves the contents of a tensor by the specified number of
    elements along the time axis (axis=1). Any elements that are moved
    outside the bounds of the tensor are dropped, with new elements padded
    with the specified value.
    Arguments:
        distance: A scalar defining the distance to shift the tensor.
        padding_value: A scalar defining the value to fill empty elements
            with. (default: 0)
    Input shape:
        nD tensor with shape: `(batch_size, time_axis, value_axis)`.
    Output shape:
        nD tensor with shape: `(batch_size, time_axis, value_axis)`.
    """

    def __init__(self,
                 distance,
                 padding_value=0,
                 **kwargs):
        super(ShiftTensor, self).__init__(**kwargs)
        self.distance = distance
        self.padding_value = padding_value

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)

        padding_amount = [self.distance, 0] if self.distance > 0 \
            else [0, -self.distance]
        padding_param = [padding_amount if i == 1 else [0, 0]
                         for i in range(len(inputs.shape))]
        padded = tf.pad(inputs,
                        padding_param,
                        constant_values=self.padding_value)

        if self.distance > 0:
            outputs = padded[:, :-self.distance]
        else:
            outputs = padded[:, -self.distance:]

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ShiftTensor, self).get_config()
        config.update({'distance': self.distance,
                       'padding_value': self.padding_value})
        return config
