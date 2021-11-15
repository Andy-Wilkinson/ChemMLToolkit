"""
This type stub file was generated by pyright.
"""

from .. import backend as F
from torch.multiprocessing import *
from .pytorch import Process
from multiprocessing import *

"""Wrapper of the multiprocessing module for multi-GPU training."""
if F.get_preferred_backend() == 'pytorch':
    ...
else:
    ...