import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.layers import OneHotEncoding


class TestOneHotEncoding(tf.test.TestCase):
    def test_call_simple(self):
        input = [
            [1, 0, 2],
            [2, 1, 0]
        ]
        expected_output = [
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        ]

        self._test_call(input, expected_output,
                        depth=4)

    def test_call_boolean(self):
        input = [
            [1, 0, 2],
            [2, 1, 0]
        ]
        expected_output = [
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, True, False]
            ],
            [
                [False, False, True, False],
                [False, True, False, False],
                [True, False, False, False]
            ]
        ]

        self._test_call(input, expected_output,
                        depth=4,
                        on_value=True,
                        off_value=False)

    def test_call_axis1(self):
        input = [
            [1, 0, 2],
            [2, 1, 0]
        ]
        expected_output = [
            [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
        ]

        self._test_call(input, expected_output,
                        depth=4,
                        axis=1)

    def _test_call(self, input, expected_output, **kwargs):
        input = np.array(input)
        oneHotEncoding = OneHotEncoding(**kwargs)
        computed_shape = oneHotEncoding.compute_output_shape(input.shape)
        output = oneHotEncoding(input)

        self.assertAllEqual(expected_output, output)
        self.assertAllEqual(output.shape, computed_shape)

    def test_serialize(self):
        model_input = tf.keras.Input(shape=[3], dtype=tf.int32)
        x = OneHotEncoding(4,
                           on_value=True,
                           off_value=False,
                           axis=0)(model_input)
        model = tf.keras.Model(model_input, x)
        config = model.get_config()

        reinitialized_model = tf.keras.Model.from_config(config)
        output = reinitialized_model(np.array([1, 0, 2]))

        self.assertAllEqual(output, [[False, True, False],
                                     [True, False, False],
                                     [False, False, True],
                                     [False, False, False]])
