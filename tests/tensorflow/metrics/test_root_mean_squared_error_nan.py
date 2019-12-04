import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.metrics import RootMeanSquaredErrorNaN


class TestRootMeanSquaredErrorNaN(tf.test.TestCase):
    y_true = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [1.0, np.nan, np.nan, 3.0, np.nan],
        [4.0, np.nan, np.nan, 6.0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan]
    ]
    y_pred = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # [0.0, ...]
        [6.0, 8.0, 8.0, 7.0, 10.0],  # [0.0, 1.0, 0.0, 4.0, 0.0]
        [1.0, 9.0, 2.0, 3.0, 9.0],  # [0.0, ...]
        [3.0, 9.0, 5.0, 9.0, 9.0],  # [1.0, nan, nan, 9.0, nan]
        [5.0, 4.0, 3.0, 2.0, 1.0]  # [nan, ...]
    ]
    expected_metric = np.sqrt(15.0 / 14.0)
    expected_metric_per_sample = (np.sqrt(1.0) + np.sqrt(5.0)) / 4.0

    def test_metric(self):
        metric_obj = RootMeanSquaredErrorNaN()
        metric = metric_obj(self.y_true, self.y_pred)
        self.assertAllClose(self.expected_metric, metric)

    def test_metric_per_sample(self):
        metric_obj = RootMeanSquaredErrorNaN(per_sample=True)
        metric = metric_obj(self.y_true, self.y_pred)
        self.assertAllClose(self.expected_metric_per_sample, metric)
