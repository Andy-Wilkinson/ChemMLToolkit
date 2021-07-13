"""
This type stub file was generated by pyright.
"""

"""
molvs.validations
~~~~~~~~~~~~~~~~~

This module contains all the built-in :class:`Validations <molvs.validations.Validation>`.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
class Validation:
    """The base class that all :class:`~molvs.validations.Validation` subclasses must inherit from."""
    def __init__(self, log) -> None:
        ...
    
    def __call__(self, mol): # -> None:
        ...
    
    def run(self, mol):
        """"""
        ...
    


class SmartsValidation(Validation):
    """Abstract superclass for :class:`Validations <molvs.validations.Validation>` that log a message if a SMARTS
    pattern matches the molecule.

    Subclasses can override the following attributes:
    """
    level = ...
    message = ...
    entire_fragment = ...
    def __init__(self, log) -> None:
        ...
    
    @property
    def smarts(self):
        """The SMARTS pattern as a string. Subclasses must implement this."""
        ...
    
    def run(self, mol): # -> None:
        ...
    


class IsNoneValidation(Validation):
    """Logs an error if ``None`` is passed to the Validator.

    This can happen if RDKit failed to parse an input format. If the molecule is ``None``, no subsequent validations
    will run.
    """
    def run(self, mol): # -> None:
        ...
    


class NoAtomValidation(Validation):
    """Logs an error if the molecule has zero atoms.

    If the molecule has no atoms, no subsequent validations will run.
    """
    def run(self, mol): # -> None:
        ...
    


class DichloroethaneValidation(SmartsValidation):
    """Logs if 1,2-dichloroethane is present.

    This is provided as an example of how to subclass :class:`~molvs.validations.SmartsValidation` to check for the
    presence of a substructure.
    """
    level = ...
    smarts = ...
    entire_fragment = ...
    message = ...


class FragmentValidation(Validation):
    """Logs if certain fragments are present.

    Subclass and override the ``fragments`` class attribute to customize the list of
    :class:`FragmentPatterns <molvs.fragment.FragmentPattern>`.
    """
    fragments = ...
    def run(self, mol): # -> None:
        ...
    


class NeutralValidation(Validation):
    """Logs if not an overall neutral system."""
    def run(self, mol): # -> None:
        ...
    


class IsotopeValidation(Validation):
    """Logs if molecule contains isotopes."""
    def run(self, mol): # -> None:
        ...
    


VALIDATIONS = ...