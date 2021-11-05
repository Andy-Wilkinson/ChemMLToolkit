"""
This type stub file was generated by pyright.
"""

"""Functions used by server."""
def start_server(server_id, ip_config, num_servers, num_clients, server_state, max_queue_size=..., net_type=...):
    """Start DGL server, which will be shared with all the rpc services.

    This is a blocking function -- it returns only when the server shutdown.

    Parameters
    ----------
    server_id : int
        Current server ID (starts from 0).
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    num_clients : int
        Total number of clients that will be connected to the server.
        Note that, we do not support dynamic connection for now. It means
        that when all the clients connect to server, no client will can be added
        to the cluster.
    server_state : ServerSate object
        Store in main data used by server.
    max_queue_size : int
        Maximal size (bytes) of server queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound because DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str
        Networking type. Current options are: 'socket'.
    """
    ...

