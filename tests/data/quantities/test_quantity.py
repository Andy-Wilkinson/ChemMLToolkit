from typing import Any, Optional, Tuple, Union
import pytest
import chemmltoolkit.data.quantities as quant


class TestQuantity(object):
    def test_new(self):
        q = quant.Quantity(1.2, 3.4, False, True, 0.2)

        assert q.val_min == 1.2
        assert q.val_max == 3.4
        assert q.eq_min is False
        assert q.eq_max is True
        assert q.error == 0.2

    def test_new_exception_invalid_range(self):
        with pytest.raises(ValueError):
            _ = quant.Quantity(3.4, 1.2, False, True, 0.2)

    @pytest.mark.parametrize("operator,value", [
        (None, 4.2),
        ('>', 4.2),
        ('<', 4.2),
        ('>=', 4.2),
        ('<=', 4.2),
        ('~', 4.2),
        ('-', (3.2, 5.2))
    ])
    def test_from_value(self, operator: str,
                        value: Union[float, Tuple[float, float]]):
        q = quant.Quantity.from_value(value, operator)

        assert q.operator == operator
        assert q.value == value

    @ pytest.mark.parametrize("operator,value", [
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
    def test_from_value_exception_invalid_value(self,
                                                operator: Optional[str],
                                                value: Any):
        with pytest.raises(ValueError):
            _ = quant.Quantity.from_value(value, operator)

    @ pytest.mark.parametrize("operator", [
        ('X',),
        ('=',),
        ('',),
    ])
    def test_from_value_exception_invalid_operator(self, operator: str):
        with pytest.raises(ValueError):
            _ = quant.Quantity.from_value(4.2, operator)
