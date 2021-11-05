"""
This type stub file was generated by pyright.
"""

"""Module for SPMV rules."""
def gen_v2v_spmv_schedule(graph, mfunc, rfunc, src_frame, dst_frame, edge_frame, out, out_size, src_map=..., dst_map=..., edge_map=..., out_map=...): # -> None:
    """Generate v2v spmv schedule.

    Parameters
    ----------
    graph : utils.CtxCachedObject
        Function that generates immutable graph index on given context
    mfunc : list of builtin message func
        Builtin message function list
    rfunc : list of builtin reduce func
        Builtin reduce function list
    src_frame : var.Var
        Input source node features
    dst_frame : var.Var
        Input destination node features
    edge_frame : var.Var
        Input edge features
    out : var.Var
        Output node features
    out_size : int
        Number of output nodes
    src_map : utils.CtxCachedObject
        Function that generates source node id mapping array on given context
    dst_map : utils.CtxCachedObject
        Function that generates destination node id mapping array on given
        context
    edge_map : utils.CtxCachedObject
        Function that generates edge id mapping array on given context
    out_map : utils.CtxCachedObject
        Function that generates output id mapping array on given context
    """
    ...

def gen_v2e_spmv_schedule(graph, mfunc, src_frame, dst_frame, edge_frame, out, out_size, src_map=..., dst_map=..., edge_map=..., out_map=...): # -> None:
    """Generate v2e SPMV schedule

    Parameters
    ----------
    graph : utils.CtxCachedObject
        Function that generates immutable graph index on given context
    mfunc : list of builtin message func
        Builtin message function list
    src_frame : var.Var
        Input source node features
    dst_frame : var.Var
        Input destination node features
    edge_frame : var.Var
        Input edge features
    out : var.Var
        Output node features
    out_size : int
        Number of output nodes
    src_map : utils.CtxCachedObject
        Function that generates source node id mapping array on given context
    dst_map : utils.CtxCachedObject
        Function that generates destination node id mapping array on given
        context
    edge_map : utils.CtxCachedObject
        Function that generates edge id mapping array on given context
    out_map : utils.CtxCachedObject
        Function that generates output id mapping array on given context
    """
    ...

def gen_e2v_spmv_schedule(graph, rfunc, message_frame, out, out_size, edge_map=..., out_map=...): # -> None:
    """Generate e2v SPMV schedule.

    Parameters
    ----------
    graph : utils.CtxCachedObject
        Function that generates immutable graph index on given context
    rfunc : list of builtin reduce func
        Builtin reduce function list
    message_frame : var.Var
        Message features
    out : var.Var
        Output node features
    out_size : int
        Number of output nodes
    edge_map : utils.CtxCachedObject
        Function that generates edge id mapping array on given context
    out_map : utils.CtxCachedObject
        Function that generates output id mapping array on given context
    """
    ...

def build_gidx_and_mapping_graph(graph): # -> tuple[Unknown, None, Unknown]:
    """Build immutable graph index of the whole graph.

    Parameters
    ----------
    graph : GraphAdapter
        Graph

    Returns
    -------
    graph : utils.CtxCachedObject
        Function that generates a immutable graph index on given context
    edge_map : utils.CtxCachedObject
        Function that generates forward and backward edge mapping on given
        context
    nbits : int
        Number of ints needed to represent the graph
    """
    ...

def build_gidx_and_mapping_uv(edge_tuples, num_src, num_dst): # -> tuple[partial[Unknown], CtxCachedObject, Unknown]:
    """Build immutable graph index and mapping using the given (u, v) edges

    The matrix is of shape (num_src, num_dst).

    Parameters
    ---------
    edge_tuples : tuple of three utils.Index
        A tuple of (u, v, eid)
    num_src : int
        Number of source nodes.
    num_dst : int
        Number of destination nodes.

    Returns
    -------
    graph : utils.CtxCachedObject
        Function that generates a immutable graph index on given context
    edge_map : utils.CtxCachedObject
        Function that generates forward and backward edge mapping on given
        context
    nbits : int
        Number of ints needed to represent the graph
    """
    ...

def build_gidx_and_mapping_block(graph, block_id, edge_tuples=...): # -> tuple[partial[Unknown], CtxCachedObject, Unknown]:
    """Build immutable graph index and mapping for node flow

    Parameters
    ----------
    graph : NodeFlow
        The NodeFlow
    block_id : int
        the block Id
    edge_tuple :  tuple of three utils.Index
        A tuple of (u, v, eid)

    Returns
    -------
    graph : utils.CtxCachedObject
        Function that generates a immutable graph index on given context
    edge_map : utils.CtxCachedObject
        Function that generates forward and backward edge mapping on given
        context
    nbits : int
        Number of ints needed to represent the graph
    """
    ...

