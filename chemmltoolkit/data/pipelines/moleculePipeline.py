from __future__ import annotations
from typing import Callable, Iterator
from .dataPipeline import DataPipeline
from rdkit.Chem import Mol


class MoleculePipeline(DataPipeline[Mol]):
    def __init__(self, iter_fn: Callable[[], Iterator[Mol]]):
        super(MoleculePipeline, self).__init__(iter_fn)
