from __future__ import annotations
from typing import Callable, Generic, Iterable, Iterator, Tuple, Type
from typing import TypeVar, Union
import itertools
from sklearn.model_selection import train_test_split

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

    def train_test_split(self: DP,
                         test_size: Union[None, float, int] = None,
                         train_size: Union[None, float, int] = None,
                         random_state: Union[None, int] = None,
                         shuffle: bool = True
                         ) -> Tuple[DP, DP]:
        vals = list(iter(self))
        train, test = train_test_split(vals,
                                       test_size=test_size,
                                       train_size=train_size,
                                       random_state=random_state,
                                       shuffle=shuffle)

        return (self.create(lambda: iter(train)),
                self.create(lambda: iter(test)))

    @ classmethod
    def create(cls: Type[DP], iter_fn: Callable[[], Iterator[T]]) -> DP:
        return cls(iter_fn)

    @ classmethod
    def chain(cls: Type[DP], pipelines: Iterable[DP]) -> DP:
        return cls(lambda: itertools.chain.from_iterable(pipelines))
