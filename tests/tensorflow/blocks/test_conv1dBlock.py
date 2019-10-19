import tensorflow as tf
from tensorflow.keras.layers import Input
from chemmltoolkit.tensorflow.blocks import Conv1DBlock


class TestConv1DBlock(tf.test.TestCase):
    def test_correct_shape(self):
        x = Input(shape=(20, 100))
        x = Conv1DBlock([10, 20, 30], 2)(x)
        assert x.shape.as_list() == [None, 17, 30]

    def test_correct_shape_with_norm_and_dropout(self):
        x = Input(shape=(20, 100))
        x = Conv1DBlock([10, 20, 30], 2, batchnorm='norm', dropout=0.2)(x)
        assert x.shape.as_list() == [None, 17, 30]
