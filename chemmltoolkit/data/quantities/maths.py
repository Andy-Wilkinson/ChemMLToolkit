from typing import Callable, SupportsFloat
import numpy as np
from chemmltoolkit.data.quantities import Quantity
import math


def add(a: Quantity, b: Quantity) -> Quantity:
    val_min = a.val_min + b.val_min
    val_max = a.val_max + b.val_max
    eq_min = a.eq_min and b.eq_min
    eq_max = a.eq_max and b.eq_max
    error = a.error + b.error

    return Quantity(val_min, val_max, eq_min, eq_max, error)


def isclose(a: Quantity, b: Quantity,
            rel_tol: SupportsFloat = 1e-9,
            abs_tol: SupportsFloat = 0.0) -> bool:
    return math.isclose(a.val_min, b.val_min,
                        rel_tol=rel_tol, abs_tol=abs_tol) and \
        math.isclose(a.val_max, b.val_max,
                     rel_tol=rel_tol, abs_tol=abs_tol)


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
