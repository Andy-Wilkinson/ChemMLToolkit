import tensorflow as tf
from tensorflow.keras.layers import Input
from chemmltoolkit.tensorflow.blocks import RecurrentBlock


class TestRecurrentBlock(tf.test.TestCase):
    def test_correct_shape(self):
        x = Input(shape=(20, 100))
        x = RecurrentBlock('gru', [10, 30])([x, None])
        assert x.shape.as_list() == [None, 20, 30]

    def test_correct_shape_with_norm_and_dropout(self):
        x = Input(shape=(20, 100))
        x = RecurrentBlock(
            'gru', [10, 30], batchnorm='norm', dropout=0.2)([x, None])
        assert x.shape.as_list() == [None, 20, 30]

    def test_correct_shape_bidirectional(self):
        x = Input(shape=(20, 100))
        x = RecurrentBlock('gru', [10, 30], bidirectional=True)([x, None])
        assert x.shape.as_list() == [None, 20, 60]
