import pytest
import numpy as np
import chemmltoolkit.data.quantities as quant


class TestStringConv(object):
    @pytest.mark.parametrize("input_str,value,operator,units", [
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
    ])
    def test_from_string(self,
                         input_str,
                         value,
                         operator,
                         units):
        quantity = quant.from_string(input_str)

        assert np.isclose(quantity.value, value).all()
        assert quantity.operator == operator
        assert quantity.units == units

    @pytest.mark.parametrize("ouput_str,value,operator,units", [
        ('10.0 M', 10.0, None, 'M'),
        ('10.0 mM', 10.0e-3, None, 'M'),
        ('10.3 M', 10.3, None, 'M'),
        ('10.3 mM', 10.3e-3, None, 'M'),
        ('10.3 uM', 10.3e-6, None, 'M'),
        ('10.3 nM', 10.3e-9, None, 'M'),
        ('-10.3 uM', -10.3e-6, None, 'M'),
        ('>10.3 uM', 10.3e-6, '>', 'M'),
        ('<10.3 uM', 10.3e-6, '<', 'M'),
        ('>=10.3 uM', 10.3e-6, '>=', 'M'),
        ('<=10.3 uM', 10.3e-6, '<=', 'M'),
        ('~10.3 uM', 10.3e-6, '~', 'M'),
        ('9.5-10.5 uM', (9.5e-6, 10.5e-6), '-', 'M'),
    ])
    def test_to_string(self,
                       ouput_str,
                       value,
                       operator,
                       units):
        quantity = quant.Quantity(value, operator, units)

        assert quant.to_string(quantity) == ouput_str
        assert str(quantity) == ouput_str
        assert repr(quantity) == ouput_str
