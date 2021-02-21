import tensorflow as tf
from tensorflow import DType

NODE_FEATURES = 'node_features'
EDGE_FEATURES = 'edge_features'


def TensorGraph(node_features, edge_features):
    """Returns a tensor graph instance.

    This function simply returns a dictionary with the relevant structure
    to represent a graph of Tensorflow Tensors.

    Arguments:
        node_features: A tensor of node features.
        edge_features: A tensor of edge features.
    """
    return {
        NODE_FEATURES: node_features,
        EDGE_FEATURES: edge_features
    }


class TensorGraphShape:
    """Represents the shape of a TensorGraph.

    The shape of a TensorGraph encompases all the information that describes
    the graph's underlying tensors. This includes what tensors are specified,
    their dimensions, as well as how they are represented.

    Arguments:
        num_nodes: The number of nodes in the graph, or None if unknown.
        node_dims: The number of features assigned to each node.
        edge_dims: The number of features assigned to each edge.
        node_dtype: The TensorFlow datatype for the node features.
        edge_dtype: The TensorFlow datatype for the edge features.
    """

    def __init__(self,
                 num_nodes: int = None,
                 node_dims: int = 1, edge_dims: int = 1,
                 node_dtype: DType = tf.float32,
                 edge_dtype: DType = tf.float32):
        self.num_nodes = num_nodes
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.node_dtype = node_dtype
        self.edge_dtype = edge_dtype

    def get_padding_shapes(self):
        return {
            NODE_FEATURES: (self.num_nodes, self.node_dims),
            EDGE_FEATURES: (self.edge_dims, self.num_nodes, self.num_nodes)
        }

    def get_padding_values(self):
        return {
            'node_features': 0.0,
            'edge_features': 0.0
        }


def map_node_features(graph, map_fn):
    features_in = graph[NODE_FEATURES]
    features_out = map_fn(features_in)
    return {**graph, NODE_FEATURES: features_out}


def map_edge_features(graph, map_fn):
    features_in = graph[EDGE_FEATURES]
    features_out = map_fn(features_in)
    return {**graph, EDGE_FEATURES: features_out}
