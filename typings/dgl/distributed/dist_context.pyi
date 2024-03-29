"""
This type stub file was generated by pyright.
"""

from enum import Enum

"""Initialize the distributed services"""
SAMPLER_POOL = ...
NUM_SAMPLER_WORKERS = ...
INITIALIZED = ...
def set_initialized(value=...): # -> None:
    """Set the initialized state of rpc"""
    ...

def get_sampler_pool(): # -> tuple[None, Literal[0]]:
    """Return the sampler pool and num_workers"""
    ...

class MpCommand(Enum):
    """Enum class for multiprocessing command"""
    INIT_RPC = ...
    SET_COLLATE_FN = ...
    CALL_BARRIER = ...
    DELETE_COLLATE_FN = ...
    CALL_COLLATE_FN = ...
    CALL_FN_ALL_WORKERS = ...
    FINALIZE_POOL = ...


def init_process(rpc_config, mp_contexts):
    """Work loop in the worker"""
    ...

class CustomPool:
    """Customized worker pool"""
    def __init__(self, num_workers, rpc_config) -> None:
        """
        Customized worker pool init function
        """
        ...
    
    def set_collate_fn(self, func, dataloader_name): # -> None:
        """Set collate function in subprocess"""
        ...
    
    def submit_task(self, dataloader_name, args): # -> None:
        """Submit task to workers"""
        ...
    
    def submit_task_to_all_workers(self, func, args): # -> None:
        """Submit task to all workers"""
        ...
    
    def get_result(self, dataloader_name, timeout=...): # -> Any:
        """Get result from result queue"""
        ...
    
    def delete_collate_fn(self, dataloader_name): # -> None:
        """Delete collate function"""
        ...
    
    def call_barrier(self): # -> None:
        """Call barrier at all workers"""
        ...
    
    def close(self): # -> None:
        """Close worker pool"""
        ...
    
    def join(self): # -> None:
        """Join the close process of worker pool"""
        ...
    


def initialize(ip_config, num_servers=..., num_workers=..., max_queue_size=..., net_type=..., num_worker_threads=...): # -> None:
    """Initialize DGL's distributed module

    This function initializes DGL's distributed module. It acts differently in server
    or client modes. In the server mode, it runs the server code and never returns.
    In the client mode, it builds connections with servers for communication and
    creates worker processes for distributed sampling. `num_workers` specifies
    the number of sampling worker processes per trainer process.
    Users also have to provide the number of server processes on each machine in order
    to connect to all the server processes in the cluster of machines correctly.

    Parameters
    ----------
    ip_config: str
        File path of ip_config file
    num_servers : int
        The number of server processes on each machine. This argument is deprecated in DGL 0.7.0.
    num_workers: int
        Number of worker process on each machine. The worker processes are used
        for distributed sampling. This argument is deprecated in DGL 0.7.0.
    max_queue_size : int
        Maximal size (bytes) of client queue buffer (~20 GB on default).

        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str, optional
        Networking type. Currently the only valid option is ``'socket'``.

        Default: ``'socket'``
    num_worker_threads: int
        The number of threads in a worker process.

    Note
    ----
    Users have to invoke this API before any DGL's distributed API and framework-specific
    distributed API. For example, when used with Pytorch, users have to invoke this function
    before Pytorch's `pytorch.distributed.init_process_group`.
    """
    ...

def finalize_client(): # -> None:
    """Release resources of this client."""
    ...

def finalize_worker(): # -> None:
    """Finalize workers
       Python's multiprocessing pool will not call atexit function when close
    """
    ...

def join_finalize_worker(): # -> None:
    """join the worker close process"""
    ...

def is_initialized(): # -> Literal[False]:
    """Is RPC initialized?
    """
    ...

def exit_client(): # -> None:
    """Trainer exits

    This function is called automatically when a Python process exits. Normally,
    the training script does not need to invoke this function at the end.

    In the case that the training script needs to initialize the distributed module
    multiple times (so far, this is needed in the unit tests), the training script
    needs to call `exit_client` before calling `initialize` again.
    """
    ...

