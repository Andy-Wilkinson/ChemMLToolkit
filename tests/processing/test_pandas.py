import pandas as pd
from rdkit.Chem import PandasTools
import chemmltoolkit.processing.pandas #noqa


class TestPandas(object):
    def test_addfingerprint_morgan(self):
        df = pd.DataFrame({'Structure': ['CCO', 'c1ccccc1C', 'CCN(CC)CC']})
        PandasTools.AddMoleculeColumnToFrame(df, 'Structure')
        df = df.addfingerprint_morgan()

        assert sum(df.columns.str.startswith('M_')) == 2048
