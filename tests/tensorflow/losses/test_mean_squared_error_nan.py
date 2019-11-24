import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.losses import mean_squared_error_nan
from chemmltoolkit.tensorflow.losses import MeanSquaredErrorNaN


class TestMeanSquaredErrorNaN(tf.test.TestCase):
    y_true = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [1.0, np.nan, np.nan, 3.0, np.nan],
            [4.0, np.nan, np.nan, 6.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ]
    y_pred = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 8.0, 8.0, 7.0, 10.0],
        [1.0, 9.0, 2.0, 3.0, 9.0],
        [3.0, 9.0, 5.0, 9.0, 9.0],
        [5.0, 4.0, 3.0, 2.0, 1.0]
    ]
    expected_loss = [0.0, 1.0, 0.0, 5.0, 0.0]

    def test_config(self):
        loss_obj = MeanSquaredErrorNaN(
            reduction=tf.keras.losses.Reduction.NONE, name='my_loss')

        self.assertEqual(loss_obj.name, 'my_loss')
        self.assertEqual(loss_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_loss_fn(self):
        loss = mean_squared_error_nan(self.y_true, self.y_pred)
        self.assertAllEqual(self.expected_loss, loss)

    def test_loss_obj(self):
        loss_obj = MeanSquaredErrorNaN()
        loss = loss_obj.call(self.y_true, self.y_pred)
        self.assertAllEqual(self.expected_loss, loss)
