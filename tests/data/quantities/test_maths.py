import pytest
import chemmltoolkit.data.quantities as quant
from chemmltoolkit.data.quantities.errors import IncompatibleUnitsError


class TestMaths(object):
    @pytest.mark.parametrize("a,b,expected_result", [
        ('10 M', '20 M', '30 M'),
        ('10 M', '-20 M', '-10 M'),
        ('10 nM', '10 nM', '20 nM'),
        ('10 uM', '100 nM', '10.1 uM'),
        ('>10 nM', '10 nM', '>20 nM'),
        ('>10 nM', '>10 nM', '>20 nM'),
        ('<10 nM', '10 nM', '<20 nM'),
        ('<10 nM', '<10 nM', '<20 nM'),
        ('>=10 nM', '10 nM', '>=20 nM'),
        ('>=10 nM', '>=10 nM', '>=20 nM'),
        ('<=10 nM', '10 nM', '<=20 nM'),
        ('<=10 nM', '<=10 nM', '<=20 nM'),
        ('~10 nM', '10 nM', '~20 nM'),
        ('~10 nM', '~10 nM', '~20 nM'),
        ('>10 nM', '>=10 nM', '>20 nM'),
        ('<10 nM', '<=10 nM', '<20 nM'),
        ('10 nM', '2-3 nM', '12-13 nM'),
        ('10-20 nM', '2-3 nM', '12-23 nM'),
        ('>10 nM', '2-3 nM', '>12 nM'),
        ('<10 nM', '2-3 nM', '<13 nM'),
        ('>=10 nM', '2-3 nM', '>=12 nM'),
        ('<=10 nM', '2-3 nM', '<=13 nM'),
    ])
    def test_add(self, a, b, expected_result):
        a = quant.from_string(a)
        b = quant.from_string(b)
        expected_result = quant.from_string(expected_result)

        assert quant.isclose(quant.add(a, b), expected_result)
        assert quant.isclose(a + b, expected_result)

        assert quant.isclose(quant.add(b, a), expected_result)
        assert quant.isclose(b + a, expected_result)

    def test_add_exception_incompatible_units(self):
        a = quant.from_string('10 M')
        b = quant.from_string('10 g')

        with pytest.raises(IncompatibleUnitsError) as excinfo:
            _ = a + b

        assert excinfo.value.units == ['M', 'g']

    @pytest.mark.parametrize("a,b,expected_result", [
        ('10 M', '20 M', False),
        ('10 M', '10 M', True),
        ('10 nM', '10 nM', True),
        ('10 M', '10.000000001 M', True),
        ('10 nM', '10.000000001 nM', True),
        ('10 nM', '>10 nM', False),
        ('10 M', '>10.000000001 M', False),
        ('>10 nM', '>10 nM', True),
        ('>10 M', '>10.000000001 M', True),
        ('10 M', '10 g', False),
    ])
    def test_isclose(self, a, b, expected_result):
        a = quant.from_string(a)
        b = quant.from_string(b)

        assert quant.isclose(a, b) == expected_result
        assert quant.isclose(b, a) == expected_result

    @pytest.mark.parametrize("a,b,expected_result", [
        ('10 M', '10 M', True),
        ('10 M', '20 M', False),
        ('10 nM', '10 nM', True),
        ('10 M', '10 nM', False),
        ('10 uM', '10 nM', False),
        ('>10 nM', '>10 nM', True),
        ('<10 nM', '<10 nM', True),
        ('>10 nM', '<10 nM', False),
        ('10.0-10.5 nM', '10.0-10.5 nM', True),
        ('10 nM', '10.0-10.5 nM', False),
        ('10.5 nM', '10.0-10.5 nM', False),
        ('10.0-10.8 nM', '10.0-10.5 nM', False),
        ('10.2-10.5 nM', '10.0-10.5 nM', False),
        ('10 M', '10 g', False),
    ])
    def test_equality(self, a, b, expected_result):
        a = quant.from_string(a)
        b = quant.from_string(b)

        assert (a == b) == expected_result

    @pytest.mark.parametrize("a,b", [
        ('10 M', '-10 M'),
        ('10 nM', '-10 nM'),
        ('>10 nM', '<-10 nM'),
        ('<10 nM', '>-10 nM'),
        ('>=10 nM', '<=-10 nM'),
        ('<=10 nM', '>=-10 nM'),
        ('~10 nM', '~-10 nM'),
        ('2-3 nM', '-3--2 nM'),
    ])
    def test_neg(self, a, b):
        a = quant.from_string(a)
        b = quant.from_string(b)

        assert quant.isclose(quant.neg(a), b)
        assert quant.isclose(-a, b)
        assert quant.isclose(quant.neg(b), a)
        assert quant.isclose(-b, a)

    @pytest.mark.parametrize("a,b", [
        ('10 M', '10 M'),
        ('10 nM', '10 nM'),
        ('>10 nM', '>10 nM'),
        ('<10 nM', '<10 nM'),
        ('>=10 nM', '>=10 nM'),
        ('<=10 nM', '<=10 nM'),
        ('~10 nM', '~10 nM'),
        ('2-3 nM', '2-3 nM'),
    ])
    def test_pos(self, a, b):
        a = quant.from_string(a)
        b = quant.from_string(b)

        assert quant.isclose(+a, b)

    @pytest.mark.parametrize("a,b,expected_result", [
        ('30 M', '20 M', '10 M'),
        ('10 M', '-20 M', '30 M'),
        ('20 nM', '10 nM', '10 nM'),
        ('10.1 uM', '100 nM', '10 uM'),
        ('>10 nM', '5 nM', '>5 nM'),
        ('10 nM', '>5 nM', '<5 nM'),
        ('>10 nM', '<5 nM', '>5 nM'),
        ('<10 nM', '5 nM', '<5 nM'),
        ('10 nM', '<5 nM', '>5 nM'),
        ('<10 nM', '>5 nM', '<5 nM'),
        ('>=10 nM', '5 nM', '>=5 nM'),
        ('10 nM', '>=5 nM', '<=5 nM'),
        ('>=10 nM', '<=5 nM', '>=5 nM'),
        ('>=10 nM', '<5 nM', '>5 nM'),
        ('<=10 nM', '5 nM', '<=5 nM'),
        ('10 nM', '<=5 nM', '>=5 nM'),
        ('<=10 nM', '>=5 nM', '<=5 nM'),
        ('<=10 nM', '>5 nM', '<5 nM'),
        ('~10 nM', '5 nM', '~5 nM'),
        ('~10 nM', '~5 nM', '~5 nM'),
        ('10 nM', '2-3 nM', '7-8 nM'),
        ('12-13 nM', '10 nM', '2-3 nM'),
        ('10-20 nM', '2-3 nM', '7-18 nM'),
        ('>10 nM', '2-3 nM', '>7 nM'),
        ('<10 nM', '2-3 nM', '<8 nM'),
        ('>=10 nM', '2-3 nM', '>=7 nM'),
        ('<=10 nM', '2-3 nM', '<=8 nM'),
    ])
    def test_sub(self, a, b, expected_result):
        a = quant.from_string(a)
        b = quant.from_string(b)
        expected_result = quant.from_string(expected_result)

        assert quant.isclose(quant.sub(a, b), expected_result)
        assert quant.isclose(a - b, expected_result)
