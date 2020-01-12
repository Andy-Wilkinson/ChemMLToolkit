import tensorflow as tf
from chemmltoolkit.tensorflow.processing import adjacency_ops


class TestAdjacencyOpsTf(tf.test.TestCase):
    def test_normalise_rank3(self):
        input = tf.convert_to_tensor(
            [
                [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
                [[1, 2, 2], [2, 3, 5], [2, 5, 3]],
            ], dtype=tf.float32)

        expected_result = tf.convert_to_tensor(
            [
                [[0, 0, 1], [0, 0, 1], [0.5, 0.5, 0]],
                [[0.2, 0.4, 0.4], [0.2, 0.3, 0.5], [0.2, 0.5, 0.3]],
            ], dtype=tf.float32)

        result = adjacency_ops.normalise(input)
        self.assertAllEqual(expected_result, result)

    def test_normalise_rank4(self):
        input = tf.convert_to_tensor(
            [
                [
                    [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
                    [[1, 2, 2], [2, 3, 5], [2, 5, 3]],
                ]
            ], dtype=tf.float32)

        expected_result = tf.convert_to_tensor(
            [
                [
                    [[0, 0, 1], [0, 0, 1], [0.5, 0.5, 0]],
                    [[0.2, 0.4, 0.4], [0.2, 0.3, 0.5], [0.2, 0.5, 0.3]],
                ]
            ], dtype=tf.float32)

        result = adjacency_ops.normalise(input)
        self.assertAllEqual(expected_result, result)
