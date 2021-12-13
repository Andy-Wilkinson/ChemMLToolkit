from __future__ import annotations
from typing import BinaryIO, Iterable, Union


class Mol():
    def GetBonds(self: Mol) -> Iterable[Bond]:
        ...

    def GetNumAtoms(self: Mol,
                    onlyHeavy: int = ...,
                    onlyExplicit: bool = ...) -> int:
        ...

    def GetProp(self: Mol,
                property: str) -> str:
        ...

    def HasSubstructMatch(self: Mol,
                          query: Mol,
                          recursionPossible: bool = ...,
                          useChirality: bool = ...,
                          useQueryQueryMatches: bool = ...,
                          ) -> bool:
        ...

    def SetProp(self: Mol,
                property: str,
                value: str,
                computed: bool = ...) -> None:
        ...


class Bond():
    def GetBeginAtomIdx(self: Bond) -> int:
        ...

    def GetEndAtomIdx(self: Bond) -> int:
        ...


class ForwardSDMolSupplier(Iterable[Mol]):
    def __init__(self: ForwardSDMolSupplier,
                 fileobj: Union[str, BinaryIO],
                 sanitize: bool = ...,
                 removeHs: bool = ...,
                 strictParsing: bool = ...,
                 ) -> None:
        ...


class SmilesParserParams():
    ...


class SmilesWriteParams():
    ...


def MolFromMolBlock(molBlock: str,
                    sanitize: bool = ...,
                    removeHs: bool = ...,
                    strictParsing: bool = ...) -> Mol:
    ...


def MolFromPDBBlock(molBlock: str,
                    sanitize: bool = ...,
                    removeHs: bool = ...,
                    flavor: int = ...,
                    proximityBonding: bool = ...) -> Mol:
    ...


def MolFromSmiles(smiles: str,
                  params: SmilesParserParams = ...) -> Mol:
    ...


def MolToSmiles(mol: Mol,
                params: SmilesWriteParams = ...) -> str:
    ...
