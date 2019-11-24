import tensorflow as tf


# @tf.keras.utils.register_keras_serializable(package='ChemMLToolkit')
@tf.function
def mean_squared_error_nan(y_true, y_pred):
    """Computes the mean squared error, masking any NaN values in `y_true`.
    Arguments:
        y_true: Tensor of true targets (possibly including NaN values).
        y_pred: Tensor of predicted targets.
    Returns:
        Tensor with one scalar loss entry per sample.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    mask = ~tf.math.is_nan(y_true)
    y_true = tf.ragged.boolean_mask(y_true, mask)
    y_pred = tf.ragged.boolean_mask(y_pred, mask)

    loss = tf.reduce_mean((y_pred - y_true) ** 2, axis=-1)
    loss = tf.where(tf.math.is_nan(loss), 0.0, loss)
    return loss


mse_nan = MSE_nan = mean_squared_error_nan


# @tf.keras.utils.register_keras_serializable(package='ChemMLToolkit')
class MeanSquaredErrorNaN(tf.keras.losses.Loss):
    """Computes the mean squared error, masking any NaN values in `y_true`.
    Arguments:
        reduction: (Optional) Type of loss reduction to apply to loss. The
            default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="mean_squared_error_nan_loss"):
        super(MeanSquaredErrorNaN, self).__init__(
            reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return mean_squared_error_nan(y_true, y_pred)
