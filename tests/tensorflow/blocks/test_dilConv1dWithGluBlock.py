import tensorflow as tf
from tensorflow.keras.layers import Input
from chemmltoolkit.tensorflow.blocks import DilConv1DWithGLUBlock


class TestDilConv1DWithGLUBlock(tf.test.TestCase):
    def test_correct_shape(self):
        x = Input(shape=(20, 30))
        x = DilConv1DWithGLUBlock([30, 30, 30])(x)
        assert x.shape.as_list() == [None, 20, 30]

    def test_correct_shape_with_dropout(self):
        x = Input(shape=(20, 30))
        x = DilConv1DWithGLUBlock(
            [30, 30, 30], dropout=0.2)(x)
        assert x.shape.as_list() == [None, 20, 30]
