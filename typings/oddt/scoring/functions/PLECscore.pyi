"""
This type stub file was generated by pyright.
"""

from oddt.scoring import scorer

class PLECscore(scorer):
    def __init__(self, protein=..., n_jobs=..., version=..., depth_protein=..., depth_ligand=..., size=...) -> None:
        """PLECscore - a novel scoring function based on PLEC fingerprints. The
        underlying model can be one of:
            * linear regression
            * neural network (dense, 200x200x200)
            * random forest (100 trees)
        The scoring function is trained on PDBbind v2016 database and even with
        linear model outperforms other machine-learning ones in terms of Pearson
        correlation coefficient on "core set". For details see PLEC publication.
        PLECscore predicts binding affinity (pKi/d).

        .. versionadded:: 0.6

        Parameters
        ----------
        protein : oddt.toolkit.Molecule object
            Receptor for the scored ligands

        n_jobs: int (default=-1)
            Number of cores to use for scoring and training. By default (-1)
            all cores are allocated.

        version: str (default='linear')
            A version of scoring function ('linear', 'nn' or 'rf') - which
            model should be used for the scoring function.

        depth_protein: int (default=5)
            The depth of ECFP environments generated on the protein side of
            interaction. By default 6 (0 to 5) environments are generated.

        depth_ligand: int (default=1)
            The depth of ECFP environments generated on the ligand side of
            interaction. By default 2 (0 to 1) environments are generated.

        size: int (default=65536)
            The final size of a folded PLEC fingerprint. This setting is not
            used to limit the data encoded in PLEC fingerprint (for that
            tune the depths), but only the final lenght. Setting it to too
            low value will lead to many collisions.

        """
        ...
    
    def gen_training_data(self, pdbbind_dir, pdbbind_versions=..., home_dir=..., use_proteins=...): # -> None:
        ...
    
    def gen_json(self, home_dir=..., pdbbind_version=...): # -> str:
        ...
    
    def train(self, home_dir=..., sf_pickle=..., pdbbind_version=..., ignore_json=...): # -> str:
        ...
    
    @classmethod
    def load(self, filename=..., version=..., pdbbind_version=..., depth_protein=..., depth_ligand=..., size=...): # -> Any:
        ...
    

