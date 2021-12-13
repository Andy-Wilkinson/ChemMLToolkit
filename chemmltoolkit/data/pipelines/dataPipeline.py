from __future__ import annotations
from typing import Callable, Generic, Iterable, Iterator, List, Tuple, Type, overload
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

    @overload
    def train_test_split(self: DP,
                         test_size: Union[float, int] = ...,
                         val_size: None = ...,
                         random_state: Union[None, int] = ...,
                         shuffle: bool = ...
                         ) -> Tuple[DP, DP]:
        ...

    @overload
    def train_test_split(self: DP,
                         test_size: Union[float, int] = ...,
                         val_size: Union[float, int] = ...,
                         random_state: Union[None, int] = ...,
                         shuffle: bool = ...
                         ) -> Tuple[DP, DP, DP]:
        ...

    def train_test_split(self: DP,
                         test_size: Union[float, int] = 0.25,
                         val_size: Union[None, float, int] = None,
                         random_state: Union[None, int] = None,
                         shuffle: bool = True
                         ) -> Union[Tuple[DP, DP], Tuple[DP, DP, DP]]:
        all: List[T] = list(iter(self))

        train, test = train_test_split(all,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=shuffle)
        if val_size is None:
            return (self.create(lambda: iter(train)),
                    self.create(lambda: iter(test)))

        random_state = random_state + 1 if random_state else None
        if isinstance(val_size, float):
            val_size = int(len(all) * val_size)

        train, val = train_test_split(train,
                                      test_size=val_size,
                                      random_state=random_state,
                                      shuffle=shuffle)

        return (self.create(lambda: iter(train)),
                self.create(lambda: iter(val)),
                self.create(lambda: iter(test)))

    @ classmethod
    def create(cls: Type[DP], iter_fn: Callable[[], Iterator[T]]) -> DP:
        return cls(iter_fn)

    @ classmethod
    def chain(cls: Type[DP], pipelines: Iterable[DP]) -> DP:
        return cls(lambda: itertools.chain.from_iterable(pipelines))
