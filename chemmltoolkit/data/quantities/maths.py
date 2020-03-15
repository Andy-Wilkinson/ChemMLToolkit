from chemmltoolkit.data.quantities import Quantity
import math


def add(a: Quantity, b: Quantity) -> Quantity:
    def _compare_operators(set_a, set_b):
        return (a.operator in set_a and b.operator in set_b) \
            or (b.operator in set_a and a.operator in set_b)

    if a.units != b.units:
        # TODO: Throw with different units!
        pass

    if a.operator != '-' and b.operator != '-':
        value = a.value + b.value
        unit_prefix = a.unit_prefix if a.value > b.value else b.unit_prefix
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
        unit_prefix = a.unit_prefix
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

    units = a.units
    return Quantity(value, operator, unit_prefix, units)

    #         if a.operator is None and b.operator is None:
    #             operator = None
    #         elif (a.operator in ['>'] and b.operator in [None, '>', '>=']) \
    #             or (b.operator in ['>'] and a.operator in [None, '>', '>=']):
    #             operator = '>'
    #         elif (a.operator in ['>='] and b.operator in [None, '>=']) \
    #             or (b.operator in ['>='] and a.operator in [None, '>=']):
    #             operator = '>='
    #         else:
    #             raise ValueError


def isclose(a: Quantity, b: Quantity, rel_tol=1e-9, abs_tol=0.0) -> bool:
    if a.operator != b.operator \
            or a.unit_prefix != b.unit_prefix \
            or a.units != b.units:
        return False

    if a.operator == '-':
        return math.isclose(a.value[0], b.value[0],
                            rel_tol=rel_tol, abs_tol=abs_tol) and \
               math.isclose(a.value[1], b.value[1],
                            rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        return math.isclose(a.value, b.value,
                            rel_tol=rel_tol, abs_tol=abs_tol)
