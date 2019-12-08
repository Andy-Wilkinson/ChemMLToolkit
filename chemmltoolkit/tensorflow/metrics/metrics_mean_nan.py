import abc
import tensorflow as tf
from tensorflow.keras.metrics import Metric


class MeanMetricNaN(Metric):
    """ An abstract base class for metrics calculated by a mean of some error
    values. It handles the case where some 'y_true' values are NaN.
    Additionally, a 'per_sample' parameter will average over samples as well
    as overall.

    Derived classes must implement 'calculate_error' to return a calculated
    error from the true and predicted values. Optionally they can also
    implement 'transform_mean' where some transformation is applied to the
    mean values to determine the metric (this transformation is applied
    globally, or per sample based on the value of 'per_sample').
    """

    def __init__(self, per_sample=False, name='metric_nan', dtype=tf.float32):
        super(MeanMetricNaN, self).__init__(name=name, dtype=dtype)
        self.per_sample = per_sample
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    @abc.abstractmethod
    def calculate_error(self, y_true, y_pred):
        raise NotImplementedError('Must be implemented in subclasses.')

    def transform_mean(self, mean):
        return mean

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, self.dtype)

        error = self.calculate_error(y_true, y_pred)
        mask = ~tf.math.is_nan(error)
        error = tf.ragged.boolean_mask(error, mask)

        if self.per_sample:
            error = self.transform_mean(tf.reduce_mean(error, axis=-1))
            count = tf.cast(~tf.math.is_nan(error), tf.float32)
        else:
            error = tf.reduce_sum(error, axis=-1)
            count = tf.reduce_sum(tf.cast(mask, tf.float32), axis=-1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            error = error * sample_weight
            count = count * sample_weight

        error = tf.boolean_mask(error, ~tf.math.is_nan(error))

        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.reduce_sum(count))

    def result(self):
        mean = self.total / self.count

        if self.per_sample:
            return mean
        else:
            return self.transform_mean(mean)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class MeanAbsoluteErrorNaN(MeanMetricNaN):
    """Computes the mean absolute error, masking any NaN values in `y_true`.
    Arguments:
        per_sample: If false (the default), the error is calculated over all
            individual values. If true, the error is calculated for each
            sample individually, and the overall mean of these errors is
            returned.
    """
    def __init__(self,
                 per_sample=False,
                 name='mean_absolute_error_nan',
                 dtype=tf.float32):
        super(MeanAbsoluteErrorNaN, self).__init__(
            per_sample=per_sample, name=name, dtype=dtype)

    def calculate_error(self, y_true, y_pred):
        return tf.abs(y_pred - y_true)


class MeanSquaredErrorNaN(MeanMetricNaN):
    """Computes the mean squared error, masking any NaN values in `y_true`.
    Arguments:
        per_sample: If false (the default), the error is calculated over all
            individual values. If true, the error is calculated for each
            sample individually, and the overall mean of these errors is
            returned.
    """
    def __init__(self,
                 per_sample=False,
                 name='mean_squared_error_nan',
                 dtype=tf.float32):
        super(MeanSquaredErrorNaN, self).__init__(
            per_sample=per_sample, name=name, dtype=dtype)

    def calculate_error(self, y_true, y_pred):
        return (y_pred - y_true) ** 2


class RootMeanSquaredErrorNaN(MeanMetricNaN):
    """Computes the root mean squared error, masking any NaN values in
    `y_true`.
    Arguments:
        per_sample: If false (the default), the error is calculated over all
            individual values. If true, the error is calculated for each
            sample individually, and the overall mean of these errors is
            returned.
    """
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
