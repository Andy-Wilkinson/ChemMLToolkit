class Quantity:
    def __init__(self, value, operator, unit_prefix, units):
        self.value = value
        self.operator = operator
        self.unit_prefix = unit_prefix
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
            and self.unit_prefix == other.unit_prefix \
            and self.units == other.units
