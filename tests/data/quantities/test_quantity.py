import pytest
import chemmltoolkit.data.quantities as quant


class TestQuantity(object):
    @pytest.mark.parametrize("operator,value", [
        (None, 4.2),
        ('>', 4.2),
        ('<', 4.2),
        ('>=', 4.2),
        ('<=', 4.2),
        ('~', 4.2),
        ('-', (3.2, 5.2))
    ])
    def test_new(self, operator, value):
        q = quant.Quantity(value, operator)

        assert q.operator == operator
        assert q.value == value

    @pytest.mark.parametrize("operator,value", [
        (None, None),
        (None, 'XXX'),
        (None, (4.2, 5.3)),
        ('-', None),
        ('-', 'XXX'),
        ('-', 4.2),
        ('-', ('XXX', 4.2)),
        ('-', (4.2, 'XXX')),
        ('-', (4.2,)),
        ('-', (3.2, 4.2, 5.2)),
    ])
    def test_new_exception_invalid_value(self, operator, value):
        with pytest.raises(ValueError):
            _ = quant.Quantity(value, operator)

    @pytest.mark.parametrize("operator", [
        ('X'),
        ('='),
        (''),
    ])
    def test_new_exception_invalid_operator(self, operator):
        with pytest.raises(ValueError):
            _ = quant.Quantity(4.2, operator)
