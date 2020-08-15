import re
from chemmltoolkit.data.quantities import Quantity
import chemmltoolkit.data.quantities._constants as qconst


_regex_quantity = None
_regex_range = None


def from_string(input: str, return_units: bool = False) -> Quantity:
    if not _regex_quantity:
        _compile_regex()

    match = _regex_quantity.match(input)
    if match:
        operator = match.group('operator')
        value = match.group('value')
        unit_prefix = match.group('unit_prefix')
        units = match.group('units')

        operator = None if operator in ['', '='] else operator
        exponent = qconst.unit_prefix_dict[unit_prefix] if unit_prefix else 1.0
        value = float(value) * exponent

        quantity = Quantity(value, operator)
        return (quantity, units) if return_units else quantity

    match = _regex_range.match(input)
    if match:
        min_value = match.group('min_value')
        max_value = match.group('max_value')
        unit_prefix = match.group('unit_prefix')
        units = match.group('units')

        exponent = qconst.unit_prefix_dict[unit_prefix] if unit_prefix else 1.0
        min_value = float(min_value) * exponent
        max_value = float(max_value) * exponent

        quantity = Quantity((min_value, max_value), '-')
        return (quantity, units) if return_units else quantity


def to_string(q: Quantity) -> str:
    # lookup_val = abs(q.value) if q.operator != '-' else \
    #     max(abs(q.value[0]), abs(q.value[1]))
    # for unit_prefix, exponent in qconst.unit_prefix_dict.items():
    #     if lookup_val > exponent:
    #         break
    # exponent = qconst.unit_prefix_dict[unit_prefix]

    if not q.operator:
        return f'{q.value}'
    elif q.operator in qconst.operators_range:
        min_value, max_value = q.value
        return f'{min_value}{q.operator}{max_value}'
    else:
        return f'{q.operator}{q.value}'


def _compile_regex():
    global _regex_quantity, _regex_range

    unit_prefix_regex = '|'.join(qconst.unit_prefix_dict.keys())
    units_regex = '|'.join(qconst.units)
    units_full_regex = r'((?P<unit_prefix>' + unit_prefix_regex + ')' + \
                       r'(?P<units>' + units_regex + '))?'

    operator_regex = r'(?P<operator>(=|~|<|>|<=|>=)?)'
    value_regex = r'(?P<value>[-+]?[\d\.]+)'
    min_value_regex = value_regex.replace('value', 'min_value')
    max_value_regex = value_regex.replace('value', 'max_value')

    _regex_quantity = re.compile(
        f'^{operator_regex}{value_regex}\s*{units_full_regex}$')  # noqa: W605

    _regex_range = re.compile(
        f'^{min_value_regex}-{max_value_regex}' +
        f'\s*{units_full_regex}$')  # noqa: W605
