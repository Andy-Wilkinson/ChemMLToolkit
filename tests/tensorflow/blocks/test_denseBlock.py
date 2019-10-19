import tensorflow as tf
from tensorflow.keras.layers import Input
from chemmltoolkit.tensorflow.blocks import DenseBlock


class TestDenseBlock(tf.test.TestCase):
    def test_correct_shape(self):
        x = Input(shape=(100,))
        x = DenseBlock([10, 20, 30])(x)
        assert x.shape.as_list() == [None, 30]

    def test_correct_shape_with_norm_and_dropout(self):
        x = Input(shape=(100,))
        x = DenseBlock([10, 20, 30], batchnorm='norm', dropout=0.2)(x)
        assert x.shape.as_list() == [None, 30]
