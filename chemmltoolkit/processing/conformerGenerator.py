import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors


class ConformerGenerator():
    """Generates conformers for molecules.

    Calling the 'generate_conformers' method will perform a number of steps.

    * Generate a number of candidate conformers, pruning any within a
      specified RMSD
    * Minimise the conformers with the specified forcefield
    * A further pruning step to remove conformers that have converged within
      the specified RMSD

    Args:
        num_confs: The number of conformers to generate prior to minimisation.
            If this is not specified, a suitable number of conformers will
            be chosen per molecule based upon the flexibility.
        prune_rms_thresh: The RMS below which to prune similar conformers.
            Note that for speed the RMSD is calculated only for heavy atoms.
        embed_parameters: The RDKit EmbedParameters to use for generating
            conformers. By default this will be ETKDGv2. Note that for ease
            of use, the PruneRmsThresh property will always be set to the
            value of 'prune_rms_thresh'.
        force_field: The force field to use for minimisation. By default this
            will use UFF. Setting this value to 'None' will skip the
            minimisation step. Supported force fields are: 'uff', 'mmff94' or
            'mmff94s'.
    """

    def __init__(self,
                 num_confs=None,
                 prune_rms_thresh=0.35,
                 embed_parameters=None,
                 force_field='uff'):
        self.num_confs = num_confs
        self.prune_rms_thresh = prune_rms_thresh
        self.force_field = force_field

        self.embed_parameters = embed_parameters if embed_parameters \
            else AllChem.ETKDGv2()
        self.embed_parameters.PruneRmsThresh = prune_rms_thresh

    def generate_conformers(self, mol):
        """Generates conformers for the specified molecule.

        Args:
            smiles: The input molecule.

        Returns:
            A new molecule with embedded conformers.
        """
        mol = Chem.AddHs(mol)
        mol = self._embed_conformers(mol)
        energies = self._minimize_conformers(mol, self.force_field)
        mol = self._prune_conformers(mol, energies)
        return mol

    def _embed_conformers(self, mol):
        if self.num_confs:
            num_confs = self.num_confs
        else:
            n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if n_rot <= 7:
                num_confs = 50
            elif n_rot <= 12:
                num_confs = 200
            else:
                num_confs = 300

        AllChem.EmbedMultipleConfs(mol,
                                   numConfs=num_confs,
                                   params=self.embed_parameters)

        return mol

    def _minimize_conformers(self, mol, force_field):
        if force_field == 'uff':
            result = AllChem.UFFOptimizeMoleculeConfs(mol)
        elif force_field.startswith('mmff'):
            result = AllChem.MMFFOptimizeMoleculeConfs(mol,
                                                       mmffVariant=force_field)
        else:
            raise ValueError(f'Unexpected force field {force_field}')

        for conformer, (success, _) in zip(mol.GetConformers(), result):
            if success != 0:
                mol.RemoveConformer(conformer.GetId())

        return [energy for success, energy in result if success == 0]

    def _prune_conformers(self, mol, energies):
        # Only keep the lowest energy conformers within each RMSD
        # Uses the procedure of Chem. Inf. Model., 2012, 52, 1146
        conformers = mol.GetConformers()
        confIds_sorted = [conformers[i].GetId() for i in np.argsort(energies)]
        confIds_keep = []
        mol_noH = Chem.RemoveHs(mol)

        for confId in confIds_sorted:
            keep_conformer = True
            for refId in confIds_keep:
                rmsd = AllChem.GetBestRMS(mol_noH, mol_noH,
                                          prbId=confId, refId=refId)
                if rmsd < self.prune_rms_thresh:
                    keep_conformer = False
                    break

            if keep_conformer:
                confIds_keep.append(confId)

        # Return a new molecule with only the chosen conformers
        result = Chem.Mol(mol)
        result.RemoveAllConformers()

        for confId in confIds_keep:
            result.AddConformer(mol.GetConformer(confId), assignId=True)

        return result
