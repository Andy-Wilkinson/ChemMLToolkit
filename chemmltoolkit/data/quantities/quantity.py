from __future__ import annotations
from typing import Any, Optional, Tuple, Union
import numpy as np


class Quantity:
    def __init__(self, val_min: float, val_max: float,
                 eq_min: bool, eq_max: bool, error: float):
        if val_min > val_max:
            raise ValueError(
                '`val_min` must be less than or equal to `val_max`.')

        self.val_min = val_min
        self.val_max = val_max
        self.eq_min = eq_min
        self.eq_max = eq_max
        self.error = error

    @property
    def value(self) -> Union[float, Tuple[float, float]]:
        if self.val_min == self.val_max:
            return self.val_min
        if self.val_min == -np.inf:
            return self.val_max
        elif self.val_max == np.inf:
            return self.val_min
        else:
            return (self.val_min, self.val_max)

    @property
    def operator(self) -> Optional[str]:
        if self.val_min == self.val_max:
            return '~' if self.error == np.inf else None
        if self.val_min == -np.inf:
            return '<=' if self.eq_max else '<'
        elif self.val_max == np.inf:
            return '>=' if self.eq_min else '>'
        else:
            return '-'

    @staticmethod
    def from_value(value: Union[float, Tuple[float, float]],
                   operator: Optional[str]):
        if operator == '-':
            if not isinstance(value, tuple) or (len(value) != 2) \
                    or (type(value[0]) is not float) \
                    or (type(value[1]) is not float):
                raise ValueError('The `value` must be a tuple of two floats.')
            return Quantity(value[0], value[1], True, True, 0.0)
        else:
            if type(value) is not float:
                raise ValueError('The `value` must be a float.')

            if operator is None:
                return Quantity(value, value, True, True, 0.0)
            elif operator == '~':
                return Quantity(value, value, True, True, np.inf)
            elif operator == '>':
                return Quantity(value, np.inf, False, False, 0.0)
            elif operator == '<':
                return Quantity(-np.inf, value, False, False, 0.0)
            elif operator == '>=':
                return Quantity(value, np.inf, True, False, 0.0)
            elif operator == '<=':
                return Quantity(-np.inf, value, False, True, 0.0)
            else:
                raise ValueError(f'Invalid `operator` "{operator}".')

    def __str__(self):
        from chemmltoolkit.data.quantities.string_conv import to_string
        return to_string(self)

    def __repr__(self):
        from chemmltoolkit.data.quantities.string_conv import to_string
        return to_string(self)

    def __abs__(self):
        from chemmltoolkit.data.quantities.maths import absolute
        return absolute(self)

    def __add__(self, other: Quantity):
        from chemmltoolkit.data.quantities.maths import add
        return add(self, other)

    def __eq__(self, other: Any):
        if not isinstance(other, Quantity):
            return False

        return self.val_min == other.val_min \
            and self.val_max == other.val_max \
            and self.eq_min == other.eq_min \
            and self.eq_max == other.eq_max \
            and self.error == other.error

    def __ge__(self, other: Quantity):
        from chemmltoolkit.data.quantities.maths import greater_equal
        return greater_equal(self, other)

    def __gt__(self, other: Quantity):
        from chemmltoolkit.data.quantities.maths import greater
        return greater(self, other)

    def __le__(self, other: Quantity):
        from chemmltoolkit.data.quantities.maths import less_equal
        return less_equal(self, other)

    def __lt__(self, other: Quantity):
        from chemmltoolkit.data.quantities.maths import less
        return less(self, other)

    def __neg__(self):
        from chemmltoolkit.data.quantities.maths import neg
        return neg(self)

    def __pos__(self):
        return Quantity(self.val_min, self.val_max,
                        self.eq_min, self.eq_max, self.error)

    def __sub__(self, other: Quantity):
        from chemmltoolkit.data.quantities.maths import sub
        return sub(self, other)
