import pytest
import numpy as np
import chemmltoolkit.data.quantities as quant


class TestQuantitiesString(object):
    @pytest.mark.parametrize("input_str,value,operator,unit_prefix,units", [
        ('10 M', 10.0, None, '', 'M'),
        ('10 mM', 10.0e-3, None, 'm', 'M'),
        ('10.3 M', 10.3, None, '', 'M'),
        ('10.3 mM', 10.3e-3, None, 'm', 'M'),
        ('10.3 uM', 10.3e-6, None, 'u', 'M'),
        ('10.3 nM', 10.3e-9, None, 'n', 'M'),
        ('=10.3 uM', 10.3e-6, None, 'u', 'M'),
        ('+10.3 uM', 10.3e-6, None, 'u', 'M'),
        ('-10.3 uM', -10.3e-6, None, 'u', 'M'),
        ('>10.3 uM', 10.3e-6, '>', 'u', 'M'),
        ('<10.3 uM', 10.3e-6, '<', 'u', 'M'),
        ('>=10.3 uM', 10.3e-6, '>=', 'u', 'M'),
        ('<=10.3 uM', 10.3e-6, '<=', 'u', 'M'),
        ('~10.3 uM', 10.3e-6, '~', 'u', 'M'),
        ('9.5-10.5 uM', (9.5e-6, 10.5e-6), '-', 'u', 'M'),
    ])
    def test_from_string(self,
                         input_str,
                         value,
                         operator,
                         unit_prefix,
                         units):
        quantity = quant.from_string(input_str)

        assert np.isclose(quantity.value, value).all()
        assert quantity.operator == operator
        assert quantity.unit_prefix == unit_prefix
        assert quantity.units == units

    @pytest.mark.parametrize("ouput_str,value,operator,unit_prefix,units", [
        ('10.0 M', 10.0, None, '', 'M'),
        ('10.0 mM', 10.0e-3, None, 'm', 'M'),
        ('10.3 M', 10.3, None, '', 'M'),
        ('10.3 mM', 10.3e-3, None, 'm', 'M'),
        ('10.3 uM', 10.3e-6, None, 'u', 'M'),
        ('10.3 nM', 10.3e-9, None, 'n', 'M'),
        ('-10.3 uM', -10.3e-6, None, 'u', 'M'),
        ('>10.3 uM', 10.3e-6, '>', 'u', 'M'),
        ('<10.3 uM', 10.3e-6, '<', 'u', 'M'),
        ('>=10.3 uM', 10.3e-6, '>=', 'u', 'M'),
        ('<=10.3 uM', 10.3e-6, '<=', 'u', 'M'),
        ('~10.3 uM', 10.3e-6, '~', 'u', 'M'),
        ('9.5-10.5 uM', (9.5e-6, 10.5e-6), '-', 'u', 'M'),
    ])
    def test_to_string(self,
                       ouput_str,
                       value,
                       operator,
                       unit_prefix,
                       units):
        quantity = quant.Quantity(value, operator, unit_prefix, units)

        assert quant.to_string(quantity) == ouput_str
        assert str(quantity) == ouput_str
        assert repr(quantity) == ouput_str