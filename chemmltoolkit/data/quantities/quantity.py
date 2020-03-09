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
