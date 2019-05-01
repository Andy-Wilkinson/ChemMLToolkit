import tensorflow as tf

if tf.__version__.startswith('1.'):
    tf.enable_eager_execution()
