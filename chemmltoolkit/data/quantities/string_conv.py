from typing import Tuple
import re

from chemmltoolkit.data.quantities import Quantity
from chemmltoolkit.data.quantities._constants \
    import unit_prefix_dict, units


class RegexCache():
    def __init__(self):
        unit_prefix_regex = '|'.join(unit_prefix_dict.keys())
        units_regex = '|'.join(units)
        units_full_regex = r'((?P<unit_prefix>' + unit_prefix_regex + ')' + \
            r'(?P<units>' + units_regex + '))?'

        operator_regex = r'(?P<operator>(=|~|<|>|<=|>=)?)'
        value_regex = r'(?P<value>[-+]?[\d\.]+)'
        min_value_regex = value_regex.replace('value', 'min_value')
        max_value_regex = value_regex.replace('value', 'max_value')

        self.quantity = re.compile(
            rf'^{operator_regex}{value_regex}$')

        self.range = re.compile(
            rf'^{min_value_regex}-{max_value_regex}$')

        self.quantity_with_units = re.compile(
            rf'^{operator_regex}{value_regex}\s*{units_full_regex}$')

        self.range_with_units = re.compile(
            rf'^{min_value_regex}-{max_value_regex}' +
            rf'\s*{units_full_regex}$')


_regex_cache = None


def from_string(input: str, return_units: bool = False) -> Quantity:
    global _regex_cache
    if not _regex_cache:
        _regex_cache = RegexCache()

    match = _regex_cache.quantity.match(input)
    if match:
        operator = match.group('operator')
        value = float(match.group('value'))

        operator = None if operator in ['', '='] else operator
        quantity = Quantity.from_value(value, operator)
        return quantity

    match = _regex_cache.range.match(input)
    if match:
        min_value = float(match.group('min_value'))
        max_value = float(match.group('max_value'))

        quantity = Quantity(min_value, max_value, True, True, 0.0)
        return quantity

    raise ValueError(f"Input is not a valid Quantity: '{input}'")


def from_string_with_units(input: str) -> Tuple[Quantity, str]:
    global _regex_cache
    if not _regex_cache:
        _regex_cache = RegexCache()

    match = _regex_cache.quantity_with_units.match(input)
    if match:
        operator = match.group('operator')
        value = float(match.group('value'))
        unit_prefix = match.group('unit_prefix')
        units = match.group('units')

        operator = None if operator in ['', '='] else operator
        exponent = unit_prefix_dict[unit_prefix] if unit_prefix else 1.0
        value = value * exponent

        quantity = Quantity.from_value(value, operator)
        return (quantity, units)

    match = _regex_cache.range_with_units.match(input)
    if match:
        min_value = float(match.group('min_value'))
        max_value = float(match.group('max_value'))
        unit_prefix = match.group('unit_prefix')
        units = match.group('units')

        exponent = unit_prefix_dict[unit_prefix] if unit_prefix else 1.0
        min_value = min_value * exponent
        max_value = max_value * exponent

        quantity = Quantity(min_value, max_value, True, True, 0.0)
        return (quantity, units)

    raise ValueError(f"Input is not a valid Quantity: '{input}'")


def to_string(q: Quantity) -> str:
    if not q.operator:
        return f'{q.value}'
    elif isinstance(q.value, float):
        return f'{q.operator}{q.value}'
    else:
        min_value, max_value = q.value
        return f'{min_value}{q.operator}{max_value}'
