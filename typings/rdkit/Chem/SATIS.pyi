"""
This type stub file was generated by pyright.
"""

"""  Functionality for SATIS typing atoms

"""
aldehydePatt = ...
ketonePatt = ...
amidePatt = ...
esterPatt = ...
carboxylatePatt = ...
carboxylPatt = ...
specialCases = ...
def SATISTypes(mol, neighborsToInclude=...):
  """ returns SATIS codes for all atoms in a molecule

   The SATIS definition used is from:
   J. Chem. Inf. Comput. Sci. _39_ 751-757 (1999)

   each SATIS code is a string consisting of _neighborsToInclude_ + 1
   2 digit numbers

   **Arguments**

     - mol: a molecule

     - neighborsToInclude (optional): the number of neighbors to include
       in the SATIS codes

   **Returns**

     a list of strings nAtoms long

  """
  ...

