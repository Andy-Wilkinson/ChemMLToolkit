"""
This type stub file was generated by pyright.
"""

from .utils import memoized_property

"""
molvs.normalize
~~~~~~~~~~~~~~~

This module contains tools for normalizing molecules using reaction SMARTS patterns.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
log = ...
class Normalization:
    """A normalization transform defined by reaction SMARTS."""
    def __init__(self, name, transform) -> None:
        """
        :param string name: A name for this Normalization
        :param string transform: Reaction SMARTS to define the transformation.
        """
        ...
    
    @memoized_property
    def transform(self):
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __str__(self) -> str:
        ...
    


NORMALIZATIONS = ...
MAX_RESTARTS = ...
class Normalizer:
    """A class for applying Normalization transforms.

    This class is typically used to apply a series of Normalization transforms to correct functional groups and
    recombine charges. Each transform is repeatedly applied until no further changes occur.
    """
    def __init__(self, normalizations=..., max_restarts=...) -> None:
        """Initialize a Normalizer with an optional custom list of :class:`~molvs.normalize.Normalization` transforms.

        :param normalizations: A list of  :class:`~molvs.normalize.Normalization` transforms to apply.
        :param int max_restarts: The maximum number of times to attempt to apply the series of normalizations (default
                                 200).
        """
        ...
    
    def __call__(self, mol):
        """Calling a Normalizer instance like a function is the same as calling its normalize(mol) method."""
        ...
    
    def normalize(self, mol):
        """Apply a series of Normalization transforms to correct functional groups and recombine charges.

        A series of transforms are applied to the molecule. For each Normalization, the transform is applied repeatedly
        until no further changes occur. If any changes occurred, we go back and start from the first Normalization
        again, in case the changes mean an earlier transform is now applicable. The molecule is returned once the entire
        series of Normalizations cause no further changes or if max_restarts (default 200) is reached.

        :param mol: The molecule to normalize.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The normalized fragment.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
        ...
    

