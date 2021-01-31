import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraphShape


def define_tfrecord_features(shape: TensorGraphShape, name='graph'):
    return {
        f'{name}:num_nodes': tf.io.FixedLenFeature([], tf.int64),
        f'{name}:node_features': tf.io.VarLenFeature(shape.node_dtype),
        f'{name}:edge_features': tf.io.VarLenFeature(shape.edge_dtype)
    }
