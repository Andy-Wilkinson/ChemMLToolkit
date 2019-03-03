import pandas as pd
from rdkit.Chem import Mol, MolToSmiles
from rdkit.Chem import PandasTools
import chemmltoolkit.processing.pandas  # noqa


class TestPandas(object):
    def test_addmoleculecolumn(self):
        df = pd.DataFrame({'Smiles': ['CO', 'c1ccccc1', 'C[NH+](C)C.[Cl-]']})
        df.addmoleculecolumn(smilesCol='Smiles')

        assert df.columns.contains('ROMol')
        assert type(df['ROMol'].values[0]) == Mol
        assert MolToSmiles(df['ROMol'].values[0]) == 'CO'
        assert MolToSmiles(df['ROMol'].values[1]) == 'c1ccccc1'
        assert MolToSmiles(df['ROMol'].values[2]) == 'C[NH+](C)C.[Cl-]'

    def test_addmoleculecolumn_withcleaning(self):
        df = pd.DataFrame({'Smiles': ['CO', 'c1ccccc1', 'C[NH+](C)C.[Cl-]']})
        df.addmoleculecolumn(smilesCol='Smiles', clean=True)

        assert df.columns.contains('ROMol')
        assert type(df['ROMol'].values[0]) == Mol
        assert MolToSmiles(df['ROMol'].values[0]) == 'CO'
        assert MolToSmiles(df['ROMol'].values[1]) == 'c1ccccc1'
        assert MolToSmiles(df['ROMol'].values[2]) == 'CN(C)C'

    def test_addfingerprint_morgan(self):
        df = pd.DataFrame({'Structure': ['CCO', 'c1ccccc1C', 'CCN(CC)CC']})
        PandasTools.AddMoleculeColumnToFrame(df, 'Structure')
        df = df.addfingerprint_morgan()

        assert sum(df.columns.str.startswith('M_')) == 2048
