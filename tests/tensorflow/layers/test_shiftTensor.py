import numpy as np
import tensorflow as tf
from chemmltoolkit.tensorflow.layers import ShiftTensor


class TestShiftTensor(tf.test.TestCase):
    def test_call_shift_one_3d(self):
        input = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
            [[2, 4, 6], [8, 2, 4], [6, 8, 2], [4, 6, 8]],
        ]
        expected_output = [
            [[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0, 0], [2, 4, 6], [8, 2, 4], [6, 8, 2]],
        ]

        self._test_call(input, expected_output,
                        distance=1)

    def test_call_shift_one_2d(self):
        input = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
        ]
        expected_output = [
            [0, 1, 2, 3],
            [0, 2, 4, 6],
        ]

        self._test_call(input, expected_output,
                        distance=1)

    def test_call_shift_two(self):
        input = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
            [[2, 4, 6], [8, 2, 4], [6, 8, 2], [4, 6, 8]],
        ]
        expected_output = [
            [[0, 0, 0], [0, 0, 0], [1, 2, 3], [4, 5, 6]],
            [[0, 0, 0], [0, 0, 0], [2, 4, 6], [8, 2, 4]],
        ]

        self._test_call(input, expected_output,
                        distance=2)

    def test_call_shift_minusone(self):
        input = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
            [[2, 4, 6], [8, 2, 4], [6, 8, 2], [4, 6, 8]],
        ]
        expected_output = [
            [[4, 5, 6], [7, 8, 9], [1, 2, 3], [0, 0, 0]],
            [[8, 2, 4], [6, 8, 2], [4, 6, 8], [0, 0, 0]],
        ]

        self._test_call(input, expected_output,
                        distance=-1)

    def test_call_shift_withpadding(self):
        input = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
            [[2, 4, 6], [8, 2, 4], [6, 8, 2], [4, 6, 8]],
        ]
        expected_output = [
            [[5, 5, 5], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[5, 5, 5], [2, 4, 6], [8, 2, 4], [6, 8, 2]],
        ]

        self._test_call(input, expected_output,
                        distance=1,
                        padding_value=5)

    def _test_call(self, input, expected_output, **kwargs):
        input = np.array(input)
        shiftTensor = ShiftTensor(**kwargs)
        computed_shape = shiftTensor.compute_output_shape(input.shape)
        output = shiftTensor(input)

        self.assertAllEqual(expected_output, output)
        self.assertAllEqual(output.shape, computed_shape)

    def test_serialize(self):
        input = np.array([[
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]],
            [[2, 4, 6], [8, 2, 4], [6, 8, 2], [4, 6, 8]],
        ]])
        expected_output = [[
            [[5, 5, 5], [5, 5, 5], [1, 2, 3], [4, 5, 6]],
            [[5, 5, 5], [5, 5, 5], [2, 4, 6], [8, 2, 4]],
        ]]

        model_input = tf.keras.Input(shape=input.shape)
        x = ShiftTensor(2, padding_value=5)(model_input)
        model = tf.keras.Model(model_input, x)
        config = model.get_config()

        reinitialized_model = tf.keras.Model.from_config(config)
        output = reinitialized_model(input)

        self.assertAllEqual(expected_output, output)
