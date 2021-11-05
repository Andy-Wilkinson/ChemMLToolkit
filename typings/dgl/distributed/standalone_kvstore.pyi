"""
This type stub file was generated by pyright.
"""

"""Define a fake kvstore

This kvstore is used when running in the standalone mode
"""
class KVClient:
    ''' The fake KVStore client.

    This is to mimic the distributed KVStore client. It's used for DistGraph
    in standalone mode.
    '''
    def __init__(self) -> None:
        ...
    
    @property
    def all_possible_part_policy(self): # -> dict[Unknown, Unknown]:
        """Get all possible partition policies"""
        ...
    
    @property
    def num_servers(self): # -> Literal[1]:
        """Get the number of servers"""
        ...
    
    def barrier(self): # -> None:
        '''barrier'''
        ...
    
    def register_push_handler(self, name, func): # -> None:
        '''register push handler'''
        ...
    
    def register_pull_handler(self, name, func): # -> None:
        '''register pull handler'''
        ...
    
    def add_data(self, name, tensor, part_policy): # -> None:
        '''add data to the client'''
        ...
    
    def init_data(self, name, shape, dtype, part_policy, init_func, is_gdata=...): # -> None:
        '''add new data to the client'''
        ...
    
    def delete_data(self, name): # -> None:
        '''delete the data'''
        ...
    
    def data_name_list(self): # -> list[Unknown]:
        '''get the names of all data'''
        ...
    
    def gdata_name_list(self): # -> list[Unknown]:
        '''get the names of graph data'''
        ...
    
    def get_data_meta(self, name): # -> tuple[Unknown, Unknown, None]:
        '''get the metadata of data'''
        ...
    
    def push(self, name, id_tensor, data_tensor): # -> None:
        '''push data to kvstore'''
        ...
    
    def pull(self, name, id_tensor):
        '''pull data from kvstore'''
        ...
    
    def map_shared_data(self, partition_book): # -> None:
        '''Mapping shared-memory tensor from server to client.'''
        ...
    


