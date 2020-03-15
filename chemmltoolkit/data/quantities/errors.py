class IncompatibleUnitsError(Exception):
    """Exception raised for calculations with incompatible units.

    Args:
        units: A list of the incompatible units.
    """

    def __init__(self, units):
        self.units = units
