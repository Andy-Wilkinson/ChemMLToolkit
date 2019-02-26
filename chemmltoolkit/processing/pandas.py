import pandas as pd
from rdkit import Chem


def df_addfingerprint_morgan(df, molCol='ROMol', prefix='M_', radius=2):
    def generate_fingerprint(mol):
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius)
        return pd.Series(list(fp))
    return df.join(df[molCol].apply(generate_fingerprint).add_prefix(prefix))


pd.DataFrame.addfingerprint_morgan = df_addfingerprint_morgan

del(df_addfingerprint_morgan)
