from __future__ import annotations
from typing import Iterable


class Mol():
    def GetBonds(self: Mol) -> Iterable[Bond]:
        ...

    def GetNumAtoms(self: Mol,
                    onlyHeavy: int = ...,
                    onlyExplicit: bool = ...) -> int:
        ...

    def HasSubstructMatch(self: Mol,
                          query: Mol,
                          recursionPossible: bool = ...,
                          useChirality: bool = ...,
                          useQueryQueryMatches: bool = ...,
                          ) -> bool:
        ...


class Bond():
    def GetBeginAtomIdx(self: Bond) -> int:
        ...

    def GetEndAtomIdx(self: Bond) -> int:
        ...


class SmilesParserParams():
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
