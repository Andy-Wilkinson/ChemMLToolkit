import pytest
import chemmltoolkit.data.quantities as quant


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
