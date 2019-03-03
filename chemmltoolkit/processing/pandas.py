import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from chemmltoolkit.processing.mols import MolCleaner


def df_addmoleculecolumn(df, smilesCol='Smiles', molCol='ROMol', clean=False):
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol, molCol)

    if clean:
        cleaner = MolCleaner()
        df[molCol] = df[molCol].apply(cleaner.clean_mol)

    return df


def df_addfingerprint_morgan(df, molCol='ROMol', prefix='M_', radius=2):
    def generate_fingerprint(mol):
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius)
        return pd.Series(list(fp))
    return df.join(df[molCol].apply(generate_fingerprint).add_prefix(prefix))


pd.DataFrame.addmoleculecolumn = df_addmoleculecolumn
pd.DataFrame.addfingerprint_morgan = df_addfingerprint_morgan

del(df_addmoleculecolumn)
del(df_addfingerprint_morgan)
