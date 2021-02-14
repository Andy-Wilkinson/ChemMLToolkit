import tensorflow as tf
from chemmltoolkit.tensorflow.graph import TensorGraph
from chemmltoolkit.tensorflow.graph import TensorGraphShape


def define_tfrecord_features(shape: TensorGraphShape, name: str = 'graph'):
    return {
        f'{name}:num_nodes': tf.io.FixedLenFeature([], tf.int64),
        f'{name}:node_features': tf.io.VarLenFeature(shape.node_dtype),
        f'{name}:edge_features': tf.io.VarLenFeature(shape.edge_dtype)
    }


def tfrecord_to_tensorgraph(example: dict, name: str = 'graph'):
    num_nodes = example[name + ':num_nodes']

    node_features = tf.sparse.to_dense(example[name + ':node_features'])
    node_features = tf.reshape(node_features, (num_nodes, -1))

    edge_features = tf.sparse.to_dense(example[name + ':edge_features'])
    edge_features = tf.reshape(edge_features, (-1, num_nodes, num_nodes))

    return TensorGraph(node_features, edge_features)
