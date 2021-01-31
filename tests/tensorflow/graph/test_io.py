import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraphShape
from chemmltoolkit.tensorflow.graph.io import define_tfrecord_features


class TestIo(tf.test.TestCase):
    def test_define_tfrecord_features(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        feat = define_tfrecord_features(shape)

        assert feat == {
            'graph:num_nodes': tf.io.FixedLenFeature([], tf.int64),
            'graph:node_features': tf.io.VarLenFeature(tf.float64),
            'graph:edge_features': tf.io.VarLenFeature(tf.float32),
        }

    def test_define_tfrecord_features_with_name(self):
        shape = TensorGraphShape(num_nodes=42,
                                 node_dims=8, edge_dims=6,
                                 node_dtype=tf.float64, edge_dtype=tf.float32)

        feat = define_tfrecord_features(shape, name='test')

        assert feat == {
            'test:num_nodes': tf.io.FixedLenFeature([], tf.int64),
            'test:node_features': tf.io.VarLenFeature(tf.float64),
            'test:edge_features': tf.io.VarLenFeature(tf.float32),
        }
