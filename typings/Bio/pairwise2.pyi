from __future__ import annotations
from typing import List, NamedTuple


class align():
    @staticmethod
    def globalxx(seqA: str, seqB: str) -> List[Alignment]:
        ...


class Alignment(NamedTuple):
    seqA: str
    seqB: str
    score: int
    start: int
    end: int
