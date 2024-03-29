"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod

class SamplerPool:
    """SamplerPool is an abstract class, in which the worker() method 
    should be implemented by users. SamplerPool will fork() N (N = num_worker)
    child processes, and each process will perform worker() method independently.
    Note that, the fork() API uses shared memory for N processes and the OS will
    perfrom copy-on-write on that only when developers write that piece of memory. 
    So fork N processes and load N copies of graph will not increase the memory overhead.

    For example, users can use this class like this:

      class MySamplerPool(SamplerPool):

          def worker(self):
              # Do anything here #

      if __name__ == '__main__':
          ...
          args = parser.parse_args()
          pool = MySamplerPool()
          pool.start(args.num_sender, args)
    """
    __metaclass__ = ABCMeta
    def start(self, num_worker, args): # -> None:
        """Start sampler pool

        Parameters
        ----------
        num_worker : int
            number of child process
        args : arguments
            any arguments passed by user
        """
        ...
    
    @abstractmethod
    def worker(self, args): # -> None:
        """User-defined function for worker

        Parameters
        ----------
        args : arguments
            any arguments passed by user 
        """
        ...
    


class SamplerSender:
    """SamplerSender for DGL distributed training.

    Users use SamplerSender to send sampled subgraphs (NodeFlow) 
    to remote SamplerReceiver. Note that, a SamplerSender can connect 
    to multiple SamplerReceiver currently. The underlying implementation 
    will send different subgraphs to different SamplerReceiver in parallel 
    via multi-threading.

    Parameters
    ----------
    namebook : dict
        IP address namebook of SamplerReceiver, where the
        key is recevier's ID (start from 0) and value is receiver's address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }

    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, namebook, net_type=...) -> None:
        ...
    
    def __del__(self): # -> None:
        """Finalize Sender
        """
        ...
    
    def send(self, nodeflow, recv_id): # -> None:
        """Send sampled subgraph (NodeFlow) to remote trainer. Note that, 
        the send() API is non-blocking and it returns immediately if the 
        underlying message queue is not full.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow
        recv_id : int
            receiver's ID
        """
        ...
    
    def batch_send(self, nf_list, id_list): # -> None:
        """Send a batch of subgraphs (Nodeflow) to remote trainer. Note that, 
        the batch_send() API is non-blocking and it returns immediately if the 
        underlying message queue is not full.

        Parameters
        ----------
        nf_list : list
            a list of NodeFlow object
        id_list : list
            a list of recv_id
        """
        ...
    
    def signal(self, recv_id): # -> None:
        """When the samplling of each epoch is finished, users can 
        invoke this API to tell SamplerReceiver that sampler has finished its job.

        Parameters
        ----------
        recv_id : int
            receiver's ID
        """
        ...
    


class SamplerReceiver:
    """SamplerReceiver for DGL distributed training.

    Users use SamplerReceiver to receive sampled subgraphs (NodeFlow) 
    from remote SamplerSender. Note that SamplerReceiver can receive messages 
    from multiple SamplerSenders concurrently by given the num_sender parameter. 
    Only when all SamplerSenders connected to SamplerReceiver successfully, 
    SamplerReceiver can start its job.

    Parameters
    ----------
    graph : DGLGraph
        The parent graph
    addr : str
        address of SamplerReceiver, e.g., '127.0.0.1:50051'
    num_sender : int
        total number of SamplerSender
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, graph, addr, num_sender, net_type=...) -> None:
        ...
    
    def __del__(self): # -> None:
        """Finalize Receiver
        """
        ...
    
    def __iter__(self): # -> Self@SamplerReceiver:
        """Sampler iterator
        """
        ...
    
    def __next__(self): # -> NodeFlow:
        """Return sampled NodeFlow object
        """
        ...
    


