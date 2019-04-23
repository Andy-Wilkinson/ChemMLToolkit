import tensorflow as tf
from chemmltoolkit.tensorflow.layers import OneHotEncoding


class TestOneHotEncoding(tf.test.TestCase):
    def test_call_simple(self):
        oneHotEncoding = OneHotEncoding(4)
        output = oneHotEncoding([1, 0, 2])
        self.assertAllEqual(output, [[0, 1, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 1, 0]])

    def test_call_boolean(self):
        oneHotEncoding = OneHotEncoding(4, on_value=True, off_value=False)
        output = oneHotEncoding([1, 0, 2])
        self.assertAllEqual(output, [[False, True, False, False],
                                     [True, False, False, False],
                                     [False, False, True, False]])

    def test_call_axis0(self):
        oneHotEncoding = OneHotEncoding(4, axis=0)
        output = oneHotEncoding([1, 0, 2])
        self.assertAllEqual(output, [[0, 1, 0],
                                     [1, 0, 0],
                                     [0, 0, 1],
                                     [0, 0, 0]])
