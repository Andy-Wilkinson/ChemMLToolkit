"""
This type stub file was generated by pyright.
"""

from . import rpc

"""Manage the roles in different clients.

Right now, the clients have different roles. Some clients work as samplers and
some work as trainers.
"""
REGISTER_ROLE = ...
REG_ROLE_MSG = ...
class RegisterRoleResponse(rpc.Response):
    """Send a confirmation signal (just a short string message)
    of RegisterRoleRequest to client.
    """
    def __init__(self, msg) -> None:
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    


class RegisterRoleRequest(rpc.Request):
    """Send client id and role to server

    Parameters
    ----------
    client_id : int
        ID of client
    role : str
        role of client
    """
    def __init__(self, client_id, machine_id, role) -> None:
        ...
    
    def __getstate__(self): # -> tuple[Unknown, Unknown, Unknown]:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    
    def process_request(self, server_state): # -> list[Unknown] | None:
        ...
    


GET_ROLE = ...
GET_ROLE_MSG = ...
class GetRoleResponse(rpc.Response):
    """Send the roles of all client processes"""
    def __init__(self, role) -> None:
        ...
    
    def __getstate__(self): # -> tuple[Unknown, str | Unknown]:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    


class GetRoleRequest(rpc.Request):
    """Send a request to get the roles of all client processes."""
    def __init__(self) -> None:
        ...
    
    def __getstate__(self): # -> str:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    
    def process_request(self, server_state): # -> GetRoleResponse:
        ...
    


PER_ROLE_RANK = ...
GLOBAL_RANK = ...
CUR_ROLE = ...
IS_STANDALONE = ...
def init_role(role): # -> None:
    """Initialize the role of the current process.

    Each process is associated with a role so that we can determine what
    function can be invoked in a process. For example, we do not allow some
    functions in sampler processes.

    The initialization includes registeration the role of the current process and
    get the roles of all client processes. It also computes the rank of all client
    processes in a deterministic way so that all clients will have the same rank for
    the same client process.
    """
    ...

def get_global_rank(): # -> Literal[0]:
    """Get the global rank

    The rank can globally identify the client process. For the client processes
    of the same role, their ranks are in a contiguous range.
    """
    ...

def get_rank(role): # -> Literal[0]:
    """Get the role-specific rank"""
    ...

def get_trainer_rank(): # -> Literal[0]:
    """Get the rank of the current trainer process.

    This function can only be called in the trainer process. It will result in
    an error if it's called in the process of other roles.
    """
    ...

def get_role(): # -> None:
    """Get the role of the current process"""
    ...

def get_num_trainers(): # -> int:
    """Get the number of trainer processes"""
    ...
