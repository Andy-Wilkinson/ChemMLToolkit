import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.metrics import MeanSquaredErrorNaN


class TestMeanSquaredErrorNaN(tf.test.TestCase):
    y_true = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [1.0, np.nan, np.nan, 3.0, np.nan],
        [4.0, np.nan, np.nan, 6.0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan]
    ]
    y_pred = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # [0.0, ...]
        [6.0, 8.0, 8.0, 7.0, 10.0],  # [0.0, 1.0, 0.0, 4.0, 0.0] = 1.0
        [1.0, 9.0, 2.0, 3.0, 9.0],  # [0.0, ...]
        [3.0, 9.0, 5.0, 9.0, 9.0],  # [1.0, nan, nan, 9.0, nan] = 5.0
        [5.0, 4.0, 3.0, 2.0, 1.0]  # [nan, ...]
    ]
    sample_weight = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_metric = 15.0 / 14.0
    expected_metric_weighted = 50.0 / 29.0
    expected_metric_per_sample = 6.0 / 4.0
    expected_metric_per_sample_weighted = 22.0 / 10.0

    def test_metric(self):
        metric_obj = MeanSquaredErrorNaN()
        metric = metric_obj(self.y_true, self.y_pred)
        self.assertAllClose(self.expected_metric, metric)

    def test_metric_weighted(self):
        metric_obj = MeanSquaredErrorNaN()
        metric = metric_obj(self.y_true, self.y_pred,
                            sample_weight=self.sample_weight)
        self.assertAllClose(self.expected_metric_weighted, metric)

    def test_metric_per_sample(self):
        metric_obj = MeanSquaredErrorNaN(per_sample=True)
        metric = metric_obj(self.y_true, self.y_pred)
        self.assertAllClose(self.expected_metric_per_sample, metric)

    def test_metric_per_sample_weighted(self):
        metric_obj = MeanSquaredErrorNaN(per_sample=True)
        metric = metric_obj(self.y_true, self.y_pred,
                            sample_weight=self.sample_weight)
        self.assertAllClose(self.expected_metric_per_sample_weighted, metric)
