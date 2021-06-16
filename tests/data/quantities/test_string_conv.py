from typing import Optional
import pytest
import numpy as np
import chemmltoolkit.data.quantities as quant


class TestStringConv(object):
    parameters_from_string = [
        ('10', 10.0, None, None),
        ('10.3', 10.3, None, None),
        ('=10.3', 10.3, None, None),
        ('+10.3', 10.3, None, None),
        ('-10.3', -10.3, None, None),
        ('>10.3', 10.3, '>', None),
        ('<10.3', 10.3, '<', None),
        ('>=10.3', 10.3, '>=', None),
        ('<=10.3', 10.3, '<=', None),
        ('~10.3', 10.3, '~', None),
        ('9.5-10.5', (9.5, 10.5), '-', None),
        ('10 M', 10.0, None, 'M'),
        ('10 mM', 10.0e-3, None, 'M'),
        ('10.3 M', 10.3, None, 'M'),
        ('10.3 mM', 10.3e-3, None, 'M'),
        ('10.3 uM', 10.3e-6, None, 'M'),
        ('10.3 nM', 10.3e-9, None, 'M'),
        ('=10.3 uM', 10.3e-6, None, 'M'),
        ('+10.3 uM', 10.3e-6, None, 'M'),
        ('-10.3 uM', -10.3e-6, None, 'M'),
        ('>10.3 uM', 10.3e-6, '>', 'M'),
        ('<10.3 uM', 10.3e-6, '<', 'M'),
        ('>=10.3 uM', 10.3e-6, '>=', 'M'),
        ('<=10.3 uM', 10.3e-6, '<=', 'M'),
        ('~10.3 uM', 10.3e-6, '~', 'M'),
        ('9.5-10.5 uM', (9.5e-6, 10.5e-6), '-', 'M'),
    ]

    @pytest.mark.parametrize("input_str,value,operator,units",
                             parameters_from_string)
    def test_from_string(self,
                         input_str: str,
                         value: float,
                         operator: Optional[str],
                         units: str):
        quantity = quant.from_string(input_str)

        assert np.isclose(quantity.value, value).all()
        assert quantity.operator == operator

    @pytest.mark.parametrize("input_str,value,operator,units",
                             parameters_from_string)
    def test_from_string_with_units(self,
                                    input_str: str,
                                    value: float,
                                    operator: Optional[str],
                                    units: str):
        quantity, result_units = quant.from_string(
            input_str, return_units=True)

        assert np.isclose(quantity.value, value).all()
        assert quantity.operator == operator
        assert result_units == units

    @pytest.mark.parametrize("ouput_str,value,operator", [
        ('10.0', 10.0, None),
        ('0.01', 10.0e-3, None),
        ('10.3', 10.3, None),
        ('0.0103', 10.3e-3, None),
        ('-10.3', -10.3, None),
        ('>10.3', 10.3, '>'),
        ('<10.3', 10.3, '<'),
        ('>=10.3', 10.3, '>='),
        ('<=10.3', 10.3, '<='),
        ('~10.3', 10.3, '~'),
        ('9.5-10.5', (9.5, 10.5), '-'),
    ])
    def test_to_string(self,
                       ouput_str: str,
                       value: float,
                       operator: Optional[str]):
        quantity = quant.Quantity.from_value(value, operator)

        assert quant.to_string(quantity) == ouput_str
        assert str(quantity) == ouput_str
        assert repr(quantity) == ouput_str
