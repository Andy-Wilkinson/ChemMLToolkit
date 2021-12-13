from __future__ import annotations
from typing import BinaryIO, Callable, Iterable, Iterator, Union
from .dataPipeline import DataPipeline
from rdkit.Chem import Mol, ForwardSDMolSupplier


class MoleculePipeline(DataPipeline[Mol]):
    def __init__(self, iter_fn: Callable[[], Iterator[Mol]]):
        super(MoleculePipeline, self).__init__(iter_fn)

    def filter_property(self: MoleculePipeline,
                        property: str,
                        fn: Callable[[str], bool]) -> MoleculePipeline:
        return self.filter(lambda mol: fn(mol.GetProp(property)))

    @staticmethod
    def from_list(molecules: Iterable[Mol]) -> MoleculePipeline:
        return MoleculePipeline(lambda: iter(molecules))

    @staticmethod
    def from_file_sdf(fileobj: Union[str, BinaryIO],
                      sanitize: bool = True,
                      removeHs: bool = True,
                      strictParsing: bool = True) -> MoleculePipeline:
        def _read_sdf() -> Iterator[Mol]:
            reader = ForwardSDMolSupplier(fileobj, removeHs=removeHs)
            return iter(reader)
        return MoleculePipeline(_read_sdf)
