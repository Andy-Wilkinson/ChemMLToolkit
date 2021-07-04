from typing import List, Optional, Tuple

from Bio.PDB.Atom import Atom
from . import Chain
from Bio import pairwise2
from Bio.PDB import Superimposer
from Bio.PDB.Residue import Residue as bpResidue


def align(mobile: Chain,
          reference: Chain
          ) -> Tuple[Optional[Chain], int, Optional[float]]:
    align_mobile, align_ref = _get_residue_alignments(mobile, reference)
    num_align_residues = len(align_mobile)
    if num_align_residues == 0:
        return None, 0, None

    atoms_mobile = [residue.child_dict.get('CA') for residue in align_mobile]
    atoms_ref = [residue.child_dict.get('CA') for residue in align_ref]

    atoms_paired = [(a, b) for (a, b) in zip(
        atoms_mobile, atoms_ref) if a is not None and b is not None]
    if len(atoms_paired) == 0:
        return None, 0, None
    atoms_mobile, atoms_ref = zip(*atoms_paired)

    superimposer = Superimposer()
    superimposer.set_atoms(atoms_ref, atoms_mobile)

    all_atoms_disordered: List[Atom] = []
    for residue in mobile.as_biopython().get_residues():
        # NB: Get all atoms including disordered residues
        for atom in residue.get_unpacked_list():
            all_atoms_disordered.append(atom)

    superimposer.apply(all_atoms_disordered)

    return mobile, num_align_residues, superimposer.rms


def _get_residue_alignments(chainA: Chain,
                            chainB: Chain
                            ) -> Tuple[List[bpResidue], List[bpResidue]]:
    def _seq_to_residues(chain: Chain, sequence: str):
        residues = list(chain.as_biopython().get_residues())
        offset = 0

        def _get_next_residue():
            nonlocal offset
            residue = residues[offset]
            offset += 1
            return residue

        return [None if code in ['-', '?'] else _get_next_residue()
                for code in sequence]

    seqA = chainA.get_sequence()
    seqB = chainB.get_sequence()

    alignment = pairwise2.align.globalxx(seqA, seqB)[0]

    residuesA = _seq_to_residues(chainA, alignment.seqA)
    residuesB = _seq_to_residues(chainB, alignment.seqB)

    alignA: List[bpResidue] = []
    alignB: List[bpResidue] = []

    for resA, resB in zip(residuesA, residuesB):
        if resA is not None and resB is not None:
            alignA.append(resA)
            alignB.append(resB)

    return alignA, alignB
