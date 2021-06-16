import pytest
import chemmltoolkit.data.quantities as quant


class TestMaths(object):
    @pytest.mark.parametrize("a_str,b_str,expected_result_str", [
        ('10', '20', '30'),
        ('10', '-20', '-10'),
        ('0.001', '0.002', '0.003'),
        ('>10', '10', '>20'),
        ('>10', '>10', '>20'),
        ('<10', '10', '<20'),
        ('<10', '<10', '<20'),
        ('>=10', '10', '>=20'),
        ('>=10', '>=10', '>=20'),
        ('<=10', '10', '<=20'),
        ('<=10', '<=10', '<=20'),
        ('~10', '10', '~20'),
        ('~10', '~10', '~20'),
        ('>10', '>=10', '>20'),
        ('<10', '<=10', '<20'),
        ('10', '2-3', '12-13'),
        ('10-20', '2-3', '12-23'),
        ('>10', '2-3', '>12'),
        ('<10', '2-3', '<13'),
        ('>=10', '2-3', '>=12'),
        ('<=10', '2-3', '<=13'),
    ])
    def test_add(self, a_str: str, b_str: str, expected_result_str: str):
        a = quant.from_string(a_str)
        b = quant.from_string(b_str)
        expected_result = quant.from_string(expected_result_str)

        assert quant.isclose(quant.add(a, b), expected_result)
        assert quant.isclose(a + b, expected_result)

        assert quant.isclose(quant.add(b, a), expected_result)
        assert quant.isclose(b + a, expected_result)

    @pytest.mark.parametrize("a_str,b_str,expected_result", [
        ('10', '20', False),
        ('10', '10', True),
        ('0.001', '0.001', True),
        ('10', '10.000000001', True),
        ('10', '>10', False),
        ('10', '>10.000000001', False),
        ('>10', '>10', True),
        ('>10', '>10.000000001', True),
    ])
    def test_isclose(self, a_str: str, b_str: str, expected_result: bool):
        a = quant.from_string(a_str)
        b = quant.from_string(b_str)

        assert quant.isclose(a, b) == expected_result
        assert quant.isclose(b, a) == expected_result

    @pytest.mark.parametrize("a_str,b_str,expected_result", [
        ('10', '10', True),
        ('10', '20', False),
        ('0.001', '0.001', True),
        ('>10', '>10', True),
        ('<10', '<10', True),
        ('>10', '<10', False),
        ('10.0-10.5', '10.0-10.5', True),
        ('10', '10.0-10.5', False),
        ('10.5', '10.0-10.5', False),
        ('10.0-10.8', '10.0-10.5', False),
        ('10.2-10.5', '10.0-10.5', False),
    ])
    def test_equality(self, a_str: str, b_str: str, expected_result: bool):
        a = quant.from_string(a_str)
        b = quant.from_string(b_str)

        assert (a == b) == expected_result

    @pytest.mark.parametrize("a_str,result_str", [
        ('100', '2'),
        ('0.01', '-2'),
        ('>100', '>2'),
        ('<100', '<2'),
        ('>=100', '>=2'),
        ('<=100', '<=2'),
        ('~100', '~2'),
        ('100-1000', '2-3'),
    ])
    def test_log10(self, a_str: str, result_str: str):
        a = quant.from_string(a_str)
        result = quant.from_string(result_str)

        assert quant.isclose(quant.log10(a), result)

    @pytest.mark.parametrize("a_str,b_str", [
        ('10', '-10'),
        ('0.001', '-0.001'),
        ('>10', '<-10'),
        ('<10', '>-10'),
        ('>=10', '<=-10'),
        ('<=10', '>=-10'),
        ('~10', '~-10'),
        ('2-3', '-3--2'),
    ])
    def test_neg(self, a_str: str, b_str: str):
        a = quant.from_string(a_str)
        b = quant.from_string(b_str)

        assert quant.isclose(quant.neg(a), b)
        assert quant.isclose(-a, b)
        assert quant.isclose(quant.neg(b), a)
        assert quant.isclose(-b, a)

    @pytest.mark.parametrize("a,b_str,result_str", [
        (2.0, '2', '4'),
        (3.0, '2', '9'),
        (10.0, '2', '100'),
        (10.0, '-2', '0.01'),
        (10.0, '>2', '>100'),
        (10.0, '<2', '<100'),
        (10.0, '>=2', '>=100'),
        (10.0, '<=2', '<=100'),
        (10.0, '~2', '~100'),
        (10.0, '2-3', '100-1000'),
    ])
    def test_pow(self, a: float, b_str: str, result_str: str):
        b = quant.from_string(b_str)
        result = quant.from_string(result_str)

        assert quant.isclose(quant.pow(a, b), result)

    @pytest.mark.parametrize("a_str,b_str", [
        ('10', '10'),
        ('0.001', '0.001'),
        ('>10', '>10'),
        ('<10', '<10'),
        ('>=10', '>=10'),
        ('<=10', '<=10'),
        ('~10', '~10'),
        ('2-3', '2-3'),
    ])
    def test_pos(self, a_str: str, b_str: str):
        a = quant.from_string(a_str)
        b = quant.from_string(b_str)

        assert quant.isclose(+a, b)

    @pytest.mark.parametrize("a_str,b_str,expected_result_str", [
        ('30', '20', '10'),
        ('10', '-20', '30'),
        ('0.003', '0.002', '0.001'),
        ('>10', '5', '>5'),
        ('10', '>5', '<5'),
        ('>10', '<5', '>5'),
        ('<10', '5', '<5'),
        ('10', '<5', '>5'),
        ('<10', '>5', '<5'),
        ('>=10', '5', '>=5'),
        ('10', '>=5', '<=5'),
        ('>=10', '<=5', '>=5'),
        ('>=10', '<5', '>5'),
        ('<=10', '5', '<=5'),
        ('10', '<=5', '>=5'),
        ('<=10', '>=5', '<=5'),
        ('<=10', '>5', '<5'),
        ('~10', '5', '~5'),
        ('~10', '~5', '~5'),
        ('10', '2-3', '7-8'),
        ('12-13', '10', '2-3'),
        ('10-20', '2-3', '7-18'),
        ('>10', '2-3', '>7'),
        ('<10', '2-3', '<8'),
        ('>=10', '2-3', '>=7'),
        ('<=10', '2-3', '<=8'),
    ])
    def test_sub(self, a_str: str, b_str: str, expected_result_str: str):
        a = quant.from_string(a_str)
        b = quant.from_string(b_str)
        expected_result = quant.from_string(expected_result_str)

        assert quant.isclose(quant.sub(a, b), expected_result)
        assert quant.isclose(a - b, expected_result)
