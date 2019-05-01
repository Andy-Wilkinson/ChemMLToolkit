import tensorflow as tf
from chemmltoolkit.tensorflow.callbacks import VariableScheduler


class TestVariableScheduler(tf.test.TestCase):
    def test_on_epoch_begin(self):
        def schedule(epoch, old_value):
            return old_value * epoch

        variable = tf.Variable(2.0, trainable=False)
        callback = VariableScheduler(variable, schedule)

        self.assertAllEqual(variable.read_value(), 2.0)
        callback.on_epoch_begin(1)
        self.assertAllEqual(variable.read_value(), 2.0)
        callback.on_epoch_begin(2)
        self.assertAllEqual(variable.read_value(), 4.0)
        callback.on_epoch_begin(3)
        self.assertAllEqual(variable.read_value(), 12.0)
