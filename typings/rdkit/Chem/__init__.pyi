"""
This type stub file was generated by pyright.
"""

import sys
from rdkit import DataStructs, RDConfig, rdBase
from rdkit.Geometry import rdGeometry
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import *
from rdkit.Chem.rdmolfiles import *
from rdkit.Chem.rdmolops import *
from rdkit.Chem.rdCIPLabeler import *
from rdkit.Chem.inchi import *
from rdkit.Chem.Chem import *

""" A module for molecules and stuff

 see Chem/index.html in the doc tree for documentation

"""

_HasSubstructMatchStr = ...


def QuickSmartsMatch(smi, sma, unique=..., display=...):
    ...


def CanonSmiles(smi, useChiral=...):
    ...


def SupplierFromFilename(fileN, delim=..., **kwargs):
    ...


# -> list[Unknown]:
def FindMolChiralCenters(mol, force=..., includeUnassigned=..., includeCIP=..., useLegacyImplementation=...):
    """
      >>> from rdkit import Chem
      >>> mol = Chem.MolFromSmiles('[C@H](Cl)(F)Br')
      >>> FindMolChiralCenters(mol)
      [(0, 'R')]
      >>> mol = Chem.MolFromSmiles('[C@@H](Cl)(F)Br')
      >>> FindMolChiralCenters(mol)
      [(0, 'S')]

      >>> FindMolChiralCenters(Chem.MolFromSmiles('CCC'))
      []

      By default unassigned stereo centers are not reported:

      >>> mol = Chem.MolFromSmiles('C[C@H](F)C(F)(Cl)Br')
      >>> FindMolChiralCenters(mol,force=True)
      [(1, 'S')]

      but this can be changed:

      >>> FindMolChiralCenters(mol,force=True,includeUnassigned=True)
      [(1, 'S'), (3, '?')]

      The handling of unassigned stereocenters for dependent stereochemistry is not correct 
      using the legacy implementation:

      >>> Chem.FindMolChiralCenters(Chem.MolFromSmiles('C1CC(C)C(C)C(C)C1'),includeUnassigned=True)
      [(2, '?'), (6, '?')]
      >>> Chem.FindMolChiralCenters(Chem.MolFromSmiles('C1C[C@H](C)C(C)[C@H](C)C1'),includeUnassigned=True)
      [(2, 'S'), (4, '?'), (6, 'R')]

      But works with the new implementation:

      >>> Chem.FindMolChiralCenters(Chem.MolFromSmiles('C1CC(C)C(C)C(C)C1'),includeUnassigned=True, useLegacyImplementation=False)
      [(2, '?'), (4, '?'), (6, '?')]

      Note that the new implementation also gets the correct descriptors for para-stereochemistry:

      >>> Chem.FindMolChiralCenters(Chem.MolFromSmiles('C1C[C@H](C)[C@H](C)[C@H](C)C1'),useLegacyImplementation=False)
      [(2, 'S'), (4, 's'), (6, 'R')]

      With the new implementation, if you don't care about the CIP labels of stereocenters, you can save
      some time by disabling those:

      >>> Chem.FindMolChiralCenters(Chem.MolFromSmiles('C1C[C@H](C)[C@H](C)[C@H](C)C1'), includeCIP=False, useLegacyImplementation=False)
      [(2, 'Tet_CCW'), (4, 'Tet_CCW'), (6, 'Tet_CCW')]

    """
    ...


if __name__ == '__main__':
    ...