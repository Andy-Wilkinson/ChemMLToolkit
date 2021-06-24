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
    * Align the conformers to template molecules (if specified)

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
        align_templates: A list of template molecules to align the generated
            conformers to. All matching template alignments are returned,
            however they are first pruned for similarity by
            'prune_rms_threshold' If this argument is not specified, no
            alignment is performed.
        align_rms_thresh: The RMS below which to accept alignments to
            template molecules.
    """

    def __init__(self,
                 num_confs=None,
                 prune_rms_thresh=0.35,
                 embed_parameters=None,
                 force_field='uff',
                 align_templates=None,
                 align_rms_thresh=1.0):
        self.num_confs = num_confs
        self.prune_rms_thresh = prune_rms_thresh
        self.force_field = force_field
        self.align_templates = align_templates
        self.align_rms_thresh = align_rms_thresh

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

        if self.force_field:
            energies = self._minimize_conformers(mol, self.force_field)
            mol = self._prune_conformers(mol, energies)

        if self.align_templates:
            mol = self._align_conformers(mol, self.align_templates)

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
        confIds_sorted = [conformers[int(i)].GetId()
                          for i in np.argsort(energies)]
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

    def _align_conformers(self, mol, templates):
        def _get_maps(mol, template):
            matches = mol.GetSubstructMatches(template)
            match_template = template.GetSubstructMatch(template)
            return [(template, list(zip(m, match_template))) for m in matches]

        template_maps = [_get_maps(mol, template) for template in templates]
        template_maps = [item for list in template_maps for item in list]
        template_maps = sorted(template_maps,
                               key=lambda x: len(x[1]),
                               reverse=True)

        result = Chem.Mol(mol)
        result.RemoveAllConformers()

        for conformer in mol.GetConformers():
            candidate_conformers = []
            for template, constraint_map in template_maps:
                score = AllChem.AlignMol(mol,
                                         template,
                                         prbCid=conformer.GetId(),
                                         atomMap=constraint_map)

                if score <= self.align_rms_thresh:
                    newConfId = result.AddConformer(conformer, assignId=True)
                    candidate_conformers.append(newConfId)

            accepted_conformers = candidate_conformers[:1]
            for candidate_confId in candidate_conformers[1:]:
                rmsds = [AllChem.GetConformerRMS(result,
                                                 candidate_confId,
                                                 accepted_confId,
                                                 prealigned=True)
                         for accepted_confId in accepted_conformers]
                if min(rmsds) > self.prune_rms_thresh:
                    accepted_conformers.append(candidate_confId)
                else:
                    result.RemoveConformer(candidate_confId)

        return result
