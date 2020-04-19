import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdForceFieldHelpers


MMFFGetProps = rdForceFieldHelpers.MMFFGetMoleculeProperties
MMFFGetForceField = rdForceFieldHelpers.MMFFGetMoleculeForceField


class ConformerGenerator():
    def __init__(self,
                 num_confs=None,
                 prune_rms_thresh=0.35,
                 force_field='uff'):
        self.num_confs = num_confs
        self.prune_rms_thresh = prune_rms_thresh
        self.force_field = force_field

    def generate_conformers(self, mol):
        mol = Chem.AddHs(mol)
        mol = self._embed_conformers(mol)
        energies = [self._minimize_conformer(mol, c.GetId())
                    for c in mol.GetConformers()]
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
                                   maxAttempts=1000,
                                   pruneRmsThresh=self.prune_rms_thresh)

        return mol

    def _minimize_conformer(self, mol, confId):
        if self.force_field == 'uff':
            force_field = AllChem.UFFGetMoleculeForceField(mol, confId=confId)
        elif self.force_field.startswith('mmff'):
            AllChem.MMFFSanitizeMolecule(mol)
            mmff_props = MMFFGetProps(mol, mmffVariant=self.force_field)
            force_field = MMFFGetForceField(mol, mmff_props, confId=confId)
        else:
            raise ValueError(f'Unexpected force field {self.force_field}')

        force_field.Minimize()
        energy = force_field.CalcEnergy()

        return energy

    def _prune_conformers(self, mol, energies):
        mol_noH = Chem.RemoveHs(mol)
        # Only keep the lowest energy conformers within each RMSD
        # Uses the procedure of Chem. Inf. Model., 2012, 52, 1146
        conformers = mol.GetConformers()
        confIds_sorted = [conformers[i].GetId() for i in np.argsort(energies)]
        confIds_keep = []

        for confId in confIds_sorted:
            keep_conformer = True
            for refId in confIds_keep:
                # rmsd = AllChem.GetBestRMS(mol, mol,
                #                           prbId=confId, refId=refId)
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
