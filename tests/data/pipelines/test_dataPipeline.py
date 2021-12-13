from __future__ import annotations
from typing import Callable, Iterator
from chemmltoolkit.data.pipelines import DataPipeline


class IntPipeline(DataPipeline[int]):
    def __init__(self, iter_fn: Callable[[], Iterator[int]]):
        super(IntPipeline, self).__init__(iter_fn)

    def sum_values(self) -> int:
        return sum(self)

    @staticmethod
    def from_range(start: int, end: int) -> IntPipeline:
        return IntPipeline(lambda: iter(range(start, end)))


class TestDataPipeline(object):
    def test_intpipeline_fromrange(self):
        pipeline = IntPipeline.from_range(0, 10)
        assert list(pipeline) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_intpipeline_sum_values(self):
        pipeline = IntPipeline.from_range(0, 10)
        assert pipeline.sum_values() == 45

    def test_chain(self):
        pipeline = IntPipeline.chain([
            IntPipeline.from_range(0, 3),
            IntPipeline.from_range(5, 10)])

        assert list(pipeline) == [0, 1, 2, 5, 6, 7, 8, 9]
        assert pipeline.sum_values() == 38

    def test_filter(self):
        pipeline = IntPipeline.from_range(0, 10).filter(lambda x: x % 2 == 0)
        assert list(pipeline) == [0, 2, 4, 6, 8]
        assert pipeline.sum_values() == 20

    def test_train_test_split(self):
        pipeline = IntPipeline.from_range(0, 10)

        train, test = pipeline.train_test_split(test_size=0.2)
        full_list = sorted(list(train) + list(test))

        assert len(train) == 8
        assert len(test) == 2
        assert full_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert train.sum_values() + test.sum_values() == 45

    def test_train_test_split_integer_size(self):
        pipeline = IntPipeline.from_range(0, 10)

        train, test = pipeline.train_test_split(test_size=2)
        full_list = sorted(list(train) + list(test))

        assert len(train) == 8
        assert len(test) == 2
        assert full_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert train.sum_values() + test.sum_values() == 45

    def test_train_test_split_with_validation(self):
        pipeline = IntPipeline.from_range(0, 10)

        train, val, test = pipeline.train_test_split(test_size=0.3,
                                                     val_size=0.1)
        full_list = sorted(list(train) + list(val) + list(test))

        assert len(train) == 6
        assert len(val) == 1
        assert len(test) == 3
        assert full_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert train.sum_values() + val.sum_values() + test.sum_values() == 45

    def test_train_test_split_with_validation_integer_size(self):
        pipeline = IntPipeline.from_range(0, 10)

        train, val, test = pipeline.train_test_split(test_size=3,
                                                     val_size=1)
        full_list = sorted(list(train) + list(val) + list(test))

        assert len(train) == 6
        assert len(val) == 1
        assert len(test) == 3
        assert full_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert train.sum_values() + val.sum_values() + test.sum_values() == 45
