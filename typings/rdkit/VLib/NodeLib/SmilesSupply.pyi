"""
This type stub file was generated by pyright.
"""

from rdkit.VLib.Supply import SupplyNode

class SmilesSupplyNode(SupplyNode):
    """ Smiles supplier

    Sample Usage:
      >>> import os
      >>> from rdkit import RDConfig
      >>> fileN = os.path.join(RDConfig.RDCodeDir,'VLib','NodeLib',\
                               'test_data','pgp_20.txt')
      >>> suppl = SmilesSupplyNode(fileN,delim="\\t",smilesColumn=2,nameColumn=1,titleLine=1)
      >>> ms = [x for x in suppl]
      >>> len(ms)
      20
      >>> ms[0].GetProp("_Name")
      'ALDOSTERONE'
      >>> ms[0].GetProp("ID")
      'RD-PGP-0001'
      >>> ms[1].GetProp("_Name")
      'AMIODARONE'
      >>> ms[3].GetProp("ID")
      'RD-PGP-0004'
      >>> suppl.reset()
      >>> suppl.next().GetProp("_Name")
      'ALDOSTERONE'
      >>> suppl.next().GetProp("_Name")
      'AMIODARONE'
      >>> suppl.reset()

    """
    def __init__(self, fileName, delim=..., nameColumn=..., smilesColumn=..., titleLine=..., **kwargs) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def next(self):
        """

        """
        ...
    


if __name__ == '__main__':
    ...