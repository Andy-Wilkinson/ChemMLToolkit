"""
This type stub file was generated by pyright.
"""

"""
  Generation of Murcko scaffolds from a molecule
"""
murckoTransforms = ...
def MakeScaffoldGeneric(mol):
  """ Makes a Murcko scaffold generic (i.e. all atom types->C and all bonds ->single

  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('c1ccccc1')))
  'C1CCCCC1'
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('c1ncccc1')))
  'C1CCCCC1'

  The following were associated with sf.net issue 246
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('c1[nH]ccc1')))
  'C1CCCC1'
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('C1[NH2+]C1')))
  'C1CC1'
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('C1[C@](Cl)(F)O1')))
  'CC1(C)CC1'

  """
  ...

murckoPatts = ...
murckoQ = ...
murckoQ = ...
murckoPatts = ...
aromaticNTransform = ...
def GetScaffoldForMol(mol):
  """ Return molecule object containing scaffold of mol

  >>> m = Chem.MolFromSmiles('Cc1ccccc1')
  >>> GetScaffoldForMol(m)
  <rdkit.Chem.rdchem.Mol object at 0x...>
  >>> Chem.MolToSmiles(GetScaffoldForMol(m))
  'c1ccccc1'

  >>> m = Chem.MolFromSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1')
  >>> Chem.MolToSmiles(GetScaffoldForMol(m))
  'c1ccc(Oc2ccccn2)cc1'

  """
  ...

def MurckoScaffoldSmiles(smiles=..., mol=..., includeChirality=...): # -> None:
  """ Returns MurckScaffold Smiles from smiles

  >>> MurckoScaffoldSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1')
  'c1ccc(Oc2ccccn2)cc1'

  >>> MurckoScaffoldSmiles(mol=Chem.MolFromSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1'))
  'c1ccc(Oc2ccccn2)cc1'

  """
  ...

def MurckoScaffoldSmilesFromSmiles(smiles, includeChirality=...): # -> None:
  """ Returns MurckScaffold Smiles from smiles

  >>> MurckoScaffoldSmilesFromSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1')
  'c1ccc(Oc2ccccn2)cc1'

  """
  ...

if __name__ == '__main__':
  ...
