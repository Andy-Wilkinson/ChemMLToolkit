from __future__ import annotations
from typing import Callable, Generic, Iterable, Iterator, Type, TypeVar
import itertools

T = TypeVar('T')
DP = TypeVar('DP', bound='DataPipeline')


class DataPipeline(Generic[T], Iterable[T]):
    def __init__(self, iter_fn: Callable[[], Iterator[T]]):
        self._iter_fn = iter_fn

    def __iter__(self) -> Iterator[T]:
        return self._iter_fn()

    def __len__(self) -> int:
        return len(list(iter(self)))

    def filter(self: DP, fn: Callable[[T], bool]) -> DP:
        return self.create(lambda: filter(fn, self))

    @classmethod
    def create(cls: Type[DP], iter_fn: Callable[[], Iterator[T]]) -> DP:
        return cls(iter_fn)

    @classmethod
    def chain(cls: Type[DP], pipelines: Iterable[DP]) -> DP:
        return cls(lambda: itertools.chain.from_iterable(pipelines))
