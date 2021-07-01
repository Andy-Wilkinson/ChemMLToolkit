from typing import Callable, Optional, SupportsFloat
import numpy as np
from chemmltoolkit.data.quantities import Quantity
import math


def absolute(a: Quantity) -> Quantity:
    if a.val_min < 0 and a.val_max > 0:
        abs_min = -a.val_min
        if a.val_max > abs_min:
            return Quantity(0, a.val_max, False, a.eq_max, a.error)
        else:
            return Quantity(0, abs_min, False, a.eq_min, a.error)
    else:
        abs_min = float.__abs__(a.val_min)
        abs_max = float.__abs__(a.val_max)

        if abs_min < abs_max:
            return Quantity(abs_min, abs_max, a.eq_min, a.eq_max, a.error)
        else:
            return Quantity(abs_max, abs_min, a.eq_max, a.eq_min, a.error)


def add(a: Quantity, b: Quantity) -> Quantity:
    val_min = a.val_min + b.val_min
    val_max = a.val_max + b.val_max
    eq_min = a.eq_min and b.eq_min
    eq_max = a.eq_max and b.eq_max
    error = a.error + b.error

    return Quantity(val_min, val_max, eq_min, eq_max, error)


def greater(a: Quantity, b: Quantity) -> Optional[bool]:
    if a.val_min > b.val_max:
        return True
    elif (a.val_min == b.val_max) and (a.eq_min is False or b.eq_max is False):
        return True
    elif a.val_max <= b.val_min:
        return False
    return None


def greater_equal(a: Quantity, b: Quantity) -> Optional[bool]:
    if a.val_min >= b.val_max:
        return True
    elif a.val_max < b.val_min:
        return False
    elif (a.val_max == b.val_min) and (a.eq_max is False or b.eq_min is False):
        return False
    return None


def isclose(a: Quantity, b: Quantity,
            rel_tol: SupportsFloat = 1e-9,
            abs_tol: SupportsFloat = 0.0) -> bool:
    return math.isclose(a.val_min, b.val_min,
                        rel_tol=rel_tol, abs_tol=abs_tol) and \
        math.isclose(a.val_max, b.val_max,
                     rel_tol=rel_tol, abs_tol=abs_tol)


def less(a: Quantity, b: Quantity) -> Optional[bool]:
    if a.val_max < b.val_min:
        return True
    elif (a.val_max == b.val_min) and (a.eq_max is False or b.eq_min is False):
        return True
    elif a.val_min >= b.val_max:
        return False
    return None


def less_equal(a: Quantity, b: Quantity) -> Optional[bool]:
    if a.val_max <= b.val_min:
        return True
    elif a.val_min > b.val_max:
        return False
    elif (a.val_min == b.val_max) and (a.eq_min is False or b.eq_max is False):
        return False
    return None


def log10(x: Quantity) -> Quantity:
    return map_value(x, math.log10)


def map_value(x: Quantity, fn: Callable[[float], float]) -> Quantity:
    val_min = -np.inf if x.val_min == -np.inf else fn(x.val_min)
    val_max = np.inf if x.val_max == np.inf else fn(x.val_max)
    error = x.error if x.error in [0.0, np.inf] else fn(x.error)

    return Quantity(val_min, val_max, x.eq_min, x.eq_max, error)


def neg(a: Quantity) -> Quantity:
    return Quantity(-a.val_max, -a.val_min, a.eq_max, a.eq_min, a.error)


def pow(x: float, y: Quantity) -> Quantity:
    return map_value(y, lambda y: math.pow(x, y))


def sub(a: Quantity, b: Quantity) -> Quantity:
    return a + (-b)
