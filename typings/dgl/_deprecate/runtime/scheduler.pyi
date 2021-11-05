"""
This type stub file was generated by pyright.
"""

"""For different schedulers"""
__all__ = ["schedule_send", "schedule_recv", "schedule_update_all", "schedule_snr", "schedule_apply_nodes", "schedule_apply_edges", "schedule_group_apply_edge", "schedule_push", "schedule_pull"]
def schedule_send(graph, u, v, eid, message_func, msgframe=...): # -> None:
    """Schedule send

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes
    v : utils.Index
        Destination nodes
    eid : utils.Index
        Ids of sending edges
    message_func: callable or list of callable
        The message function
    msgframe : FrameRef, optional
        The storage to write messages to. If None, use graph.msgframe.
    """
    ...

def schedule_recv(graph, recv_nodes, reduce_func, apply_func, inplace, outframe=...): # -> None:
    """Schedule recv.

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    recv_nodes: utils.Index
        Nodes to recv.
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    ...

def schedule_snr(graph, edge_tuples, message_func, reduce_func, apply_func, inplace, outframe=...): # -> None:
    """Schedule send_and_recv.

    Currently it builds a subgraph from edge_tuples with the same number of
    nodes as the original graph, so that routines for whole-graph updates
    (e.g. fused kernels) could be reused.

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    edge_tuples: tuple
        A tuple of (src ids, dst ids, edge ids) representing edges to perform
        send_and_recv
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    ...

def schedule_update_all(graph, message_func, reduce_func, apply_func, outframe=...): # -> None:
    """Get send and recv schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    ...

def schedule_apply_nodes(v, apply_func, node_frame, inplace, outframe=..., ntype=...): # -> None:
    """Get apply nodes schedule

    Parameters
    ----------
    v : utils.Index
        Nodes to apply
    apply_func : callable
        The apply node function
    node_frame : FrameRef
        Node feature frame.
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use the given node_frame.
    ntype : str, optional
        The node type, if running on a heterograph.
        If None, assuming it's running on a homogeneous graph.

    Returns
    -------
    A list of executors for DGL Runtime
    """
    ...

def schedule_nodeflow_apply_nodes(graph, layer_id, v, apply_func, inplace): # -> None:
    """Get apply nodes schedule in NodeFlow.

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    layer_id : int
        The layer where we apply node update function.
    v : utils.Index
        Nodes to apply
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place

    Returns
    -------
    A list of executors for DGL Runtime
    """
    ...

def schedule_apply_edges(graph, u, v, eid, apply_func, inplace, outframe=...): # -> None:
    """Get apply edges schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    apply_func: callable
        The apply edge function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.edge_frame.

    Returns
    -------
    A list of executors for DGL Runtime
    """
    ...

def schedule_nodeflow_apply_edges(graph, block_id, u, v, eid, apply_func, inplace): # -> None:
    """Get apply edges schedule in NodeFlow.

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    block_id : int
        The block whose edges we apply edge update function.
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    apply_func: callable
        The apply edge function
    inplace: bool
        If True, the update will be done in place

    Returns
    -------
    A list of executors for DGL Runtime
    """
    ...

def schedule_push(graph, u, message_func, reduce_func, apply_func, inplace, outframe=...): # -> None:
    """Get push schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes for push
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    ...

def schedule_pull(graph, pull_nodes, message_func, reduce_func, apply_func, inplace, outframe=...): # -> None:
    """Get pull schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    pull_nodes : utils.Index
        Destination nodes for pull
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    ...

def schedule_group_apply_edge(graph, u, v, eid, apply_func, group_by, inplace, outframe=...): # -> None:
    """Group apply edges schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    apply_func: callable
        The apply edge function
    group_by : str
        Specify how to group edges. Expected to be either 'src' or 'dst'
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.edgeframe.
    """
    ...

def schedule_nodeflow_update_all(graph, block_id, message_func, reduce_func, apply_func): # -> None:
    """Get update_all schedule in a block.

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    block_id : int
        The block where we perform computation.
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    """
    ...

def schedule_nodeflow_compute(graph, block_id, u, v, eid, dest_nodes, message_func, reduce_func, apply_func, inplace): # -> None:
    """Get flow compute schedule in NodeFlow

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    block_id : int
        The block where we perform computation.
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    dest_nodes : utils.Index
        Destination nodes ids
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    """
    ...

