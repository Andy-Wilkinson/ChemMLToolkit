import tensorflow as tf
from chemmltoolkit.tensorflow.metrics.utils import MeanMetricNaN


class MeanAbsoluteErrorNaN(MeanMetricNaN):
    def __init__(self,
                 per_sample=False,
                 name='mean_absolute_error_nan',
                 dtype=tf.float32):
        super(MeanAbsoluteErrorNaN, self).__init__(
            per_sample=per_sample, name=name, dtype=dtype)

    def calculate_error(self, y_true, y_pred):
        return tf.abs(y_pred - y_true)


class MeanSquaredErrorNaN(MeanMetricNaN):
    def __init__(self,
                 per_sample=False,
                 name='mean_squared_error_nan',
                 dtype=tf.float32):
        super(MeanSquaredErrorNaN, self).__init__(
            per_sample=per_sample, name=name, dtype=dtype)

    def calculate_error(self, y_true, y_pred):
        return (y_pred - y_true) ** 2


class RootMeanSquaredErrorNaN(MeanMetricNaN):
    def __init__(self,
                 per_sample=False,
                 name='root_mean_squared_error_nan',
                 dtype=tf.float32):
        super(RootMeanSquaredErrorNaN, self).__init__(
            per_sample=per_sample, name=name, dtype=dtype)

    def calculate_error(self, y_true, y_pred):
        return (y_pred - y_true) ** 2

    def transform_mean(self, mean):
        return tf.sqrt(mean)
