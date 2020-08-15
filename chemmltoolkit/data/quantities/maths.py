from chemmltoolkit.data.quantities import Quantity
import math


def add(a: Quantity, b: Quantity) -> Quantity:
    def _compare_operators(set_a, set_b):
        return (a.operator in set_a and b.operator in set_b) \
            or (b.operator in set_a and a.operator in set_b)

    if a.operator != '-' and b.operator != '-':
        value = a.value + b.value
        if a.operator is None and b.operator is None:
            operator = None
        elif _compare_operators(['>'], [None, '>', '>=']):
            operator = '>'
        elif _compare_operators(['>='], [None, '>=']):
            operator = '>='
        elif _compare_operators(['<'], [None, '<', '<=']):
            operator = '<'
        elif _compare_operators(['<='], [None, '<=']):
            operator = '<='
        elif a.operator == '~' or b.operator == '~':
            operator = '~'

    else:
        if a.operator != '-':
            a, b = b, a
        if b.operator is None:
            value = (a.value[0] + b.value, a.value[1] + b.value)
            operator = '-'
        elif b.operator == '-':
            value = (a.value[0] + b.value[0], a.value[1] + b.value[1])
            operator = '-'
        elif b.operator in ['>', '>=']:
            value = a.value[0] + b.value
            operator = b.operator
        elif b.operator in ['<', '<=']:
            value = a.value[1] + b.value
            operator = b.operator

    return Quantity(value, operator)


def isclose(a: Quantity, b: Quantity, rel_tol=1e-9, abs_tol=0.0) -> bool:
    if a.operator != b.operator:
        return False

    if a.operator == '-':
        return math.isclose(a.value[0], b.value[0],
                            rel_tol=rel_tol, abs_tol=abs_tol) and \
            math.isclose(a.value[1], b.value[1],
                         rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        return math.isclose(a.value, b.value,
                            rel_tol=rel_tol, abs_tol=abs_tol)


def log10(x: Quantity) -> Quantity:
    return map_value(x, math.log10)


def map_value(x: Quantity, fn) -> Quantity:
    if x.operator == '-':
        min_value, max_value = x.value
        return Quantity((fn(min_value), fn(max_value)), x.operator)
    else:
        return Quantity(fn(x.value), x.operator)


def neg(a: Quantity) -> Quantity:
    if a.operator in [None, '~']:
        return Quantity(-a.value, a.operator)
    if a.operator == '-':
        return Quantity((-a.value[1], -a.value[0]), '-')
    elif a.operator == '>':
        return Quantity(-a.value, '<')
    elif a.operator == '<':
        return Quantity(-a.value, '>')
    elif a.operator == '>=':
        return Quantity(-a.value, '<=')
    elif a.operator == '<=':
        return Quantity(-a.value, '>=')


def pow(x: float, y: Quantity) -> Quantity:
    return map_value(y, lambda y: math.pow(x, y))


def sub(a: Quantity, b: Quantity) -> Quantity:
    return a + (-b)
