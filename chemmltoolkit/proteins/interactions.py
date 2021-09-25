from typing import Generator, List, Optional, Union
import math
import numpy as np
from numpy.typing import NDArray
from . import Protein, Residue
import oddt
from oddt.interactions import hbonds


def get_covalent_residue(residue: Residue,
                         convalent_residues: List[str] = ['CYS'],
                         distance_cutoff: float = 2.0
                         ) -> Optional[Residue]:
    distance_squared = distance_cutoff ** 2

    ligand_coords = [atom.coord for atom in residue.as_biopython().get_atoms()]

    for residue_cov in residue.chain.get_residues(
            residue_names=convalent_residues):
        residue_cov_biopython = residue_cov.as_biopython()
        if residue_cov_biopython == residue.as_biopython():
            continue

        for atom in residue_cov_biopython.get_atoms():
            atom_coord = atom.coord
            dists_squared = [np.sum((c - atom_coord)**2)
                             for c in ligand_coords]
            for dist in dists_squared:
                if dist < distance_squared:
                    return residue_cov

    return None


def contact_distance(a: List[Union[Residue, NDArray[np.float32]]],
                     b: List[Union[Residue, NDArray[np.float32]]]
                     ) -> float:
    def _distance_squared(coords_a: NDArray[np.float32],
                          coords_b: NDArray[np.float32]) -> float:
        return np.sum((coords_a-coords_b)**2)

    from_coords = list(_get_atom_coordinates(a))
    return math.sqrt(min(
        min(_distance_squared(c, to_coord) for c in from_coords)
        for to_coord in _get_atom_coordinates(b)))


def get_interactions(protein: Protein,
                     ligand: oddt.toolkit.Molecule,
                     strict_only: bool = True):
    protein_oddt = protein.as_oddt()
    prot_atoms, lig_atoms, strict = hbonds(protein_oddt, ligand)
    chain = next(protein.get_chains())

    if strict_only:
        prot_atoms = prot_atoms[strict]
        lig_atoms = lig_atoms[strict]

    interactions = []

    for prot_atom, lig_atom in zip(prot_atoms, lig_atoms):
        # NB: Filter out interactions not to protein residues
        #     e.g. ligand-self, ligand-solvent
        if prot_atom['resnum'] in protein_oddt.res_dict['resnum']:
            residue = chain.get_residue(prot_atom['resnum'])

            if lig_atom['isacceptor']:
                interaction_type = 'A'
            elif lig_atom['isdonor']:
                interaction_type = 'D'
            else:
                interaction_type = 'X'

            interaction = (residue, prot_atom['isbackbone'], interaction_type)
            interactions.append(interaction)

    return interactions


def _get_atom_coordinates(items: List[Union[Residue, NDArray[np.float32]]]
                          ) -> Generator[NDArray[np.float32], None, None]:
    for item in items:
        if isinstance(item, Residue):
            for atom in item.as_biopython().get_atoms():
                yield atom.coord
        else:
            yield item
