class Quantity:
    def __init__(self, value, operator, units):
        self.value = value
        self.operator = operator
        self.units = units

    def __str__(self):
        from chemmltoolkit.data.quantities.string_conv import to_string
        return to_string(self)

    def __repr__(self):
        from chemmltoolkit.data.quantities.string_conv import to_string
        return to_string(self)

    def __add__(self, other):
        from chemmltoolkit.data.quantities.maths import add
        return add(self, other)

    def __eq__(self, other):
        if not isinstance(other, Quantity):
            return False

        return self.value == other.value \
            and self.operator == other.operator \
            and self.units == other.units

    def __neg__(self):
        from chemmltoolkit.data.quantities.maths import neg
        return neg(self)

    def __pos__(self):
        return Quantity(self.value, self.operator, self.units)

    def __sub__(self, other):
        from chemmltoolkit.data.quantities.maths import sub
        return sub(self, other)
