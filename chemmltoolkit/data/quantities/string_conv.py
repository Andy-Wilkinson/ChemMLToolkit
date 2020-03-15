import re
from chemmltoolkit.data.quantities import Quantity
import chemmltoolkit.data.quantities._constants as qconst


_regex_quantity = None
_regex_range = None


def from_string(input: str) -> Quantity:
    if not _regex_quantity:
        _compile_regex()

    match = _regex_quantity.match(input)
    if match:
        operator = match.group('operator')
        value = match.group('value')
        unit_prefix = match.group('unit_prefix')
        units = match.group('units')

        operator = None if operator in ['', '='] else operator
        exponent = qconst.unit_prefix_dict[unit_prefix]
        value = float(value) * exponent

        return Quantity(value, operator, units)

    match = _regex_range.match(input)
    if match:
        min_value = match.group('min_value')
        max_value = match.group('max_value')
        unit_prefix = match.group('unit_prefix')
        units = match.group('units')

        exponent = qconst.unit_prefix_dict[unit_prefix]
        min_value = float(min_value) * exponent
        max_value = float(max_value) * exponent

        return Quantity((min_value, max_value), '-', units)


def to_string(q: Quantity) -> str:
    lookup_val = abs(q.value) if q.operator != '-' else \
                 max(abs(q.value[0]), abs(q.value[1]))
    for unit_prefix, exponent in qconst.unit_prefix_dict.items():
        if lookup_val > exponent:
            break
    exponent = qconst.unit_prefix_dict[unit_prefix]

    if not q.operator:
        value = round(q.value / exponent, qconst.rounding_digits)
        return f'{value} {unit_prefix}{q.units}'
    elif q.operator in qconst.operators_range:
        min_value, max_value = q.value
        min_value = round(min_value / exponent, qconst.rounding_digits)
        max_value = round(max_value / exponent, qconst.rounding_digits)
        return f'{min_value}{q.operator}{max_value} {unit_prefix}{q.units}'
    else:
        value = round(q.value / exponent, qconst.rounding_digits)
        return f'{q.operator}{value} {unit_prefix}{q.units}'


def _compile_regex():
    global _regex_quantity, _regex_range

    unit_prefix_regex = '|'.join(qconst.unit_prefix_dict.keys())
    units_regex = '|'.join(qconst.units)
    units_full_regex = f'(?P<unit_prefix>{unit_prefix_regex})' + \
                       f'(?P<units>{units_regex})'

    regex = f'^(?P<operator>(=|~|<|>|<=|>=)?)' + \
            f'(?P<value>[-+]?[\d\.]+)\s*{units_full_regex}$'  # noqa: W605
    _regex_quantity = re.compile(regex)

    regex = f'^(?P<min_value>[-+]?[\d\.]+)' + \
            f'-(?P<max_value>[-+]?[\d\.]+)\s*{units_full_regex}$'  # noqa: W605
    _regex_range = re.compile(regex)
