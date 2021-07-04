from __future__ import annotations


class Mol():
    def HasSubstructMatch(self: Mol,
                          query: Mol,
                          recursionPossible: bool = ...,
                          useChirality: bool = ...,
                          useQueryQueryMatches: bool = ...,
                          ) -> bool:
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
