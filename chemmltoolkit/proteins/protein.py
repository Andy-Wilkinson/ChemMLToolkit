from __future__ import annotations
from typing import Generator, Optional, Tuple, Union
import gzip
from pathlib import Path
import Bio.PDB as biopdb
from Bio.PDB.Structure import Structure as bpStructure
from Bio.PDB.Chain import Chain as bpChain
from Bio.PDB.Residue import Residue as bpResidue
from Bio.PDB.Polypeptide import standard_aa_names
from Bio.PDB.Polypeptide import three_to_one


class Protein():
    def __init__(self, id: str):
        self.id = id
        self.filename: Optional[Path] = None
        self._biopython: Optional[bpStructure] = None

    @property
    def name(self) -> str:
        return self.id

    def as_biopython(self) -> bpStructure:
        if not self._biopython:
            if self.filename:
                file_open = gzip.open if self.filename.suffix == '.gz' \
                    else open

                with file_open(self.filename, 'rt', encoding='ascii') as file:
                    parser = biopdb.PDBParser()
                    self._biopython = parser.get_structure(None, file)
            else:
                raise Exception('No filename is specified.')

        return self._biopython

    def get_chain(self, chain_id: str) -> Chain:
        return Chain(self, chain_id)

    def get_chains(self) -> Generator[Chain, None, None]:
        chains = self.as_biopython().get_chains()
        return (Chain(self, c.id, biopython=c) for c in chains)

    def save(self, filename: Union[str, Path]):
        io = biopdb.PDBIO()
        io.set_structure(self.as_biopython())
        io.save(str(filename))

    @staticmethod
    def from_file(filename: Union[str, Path]) -> Protein:
        if isinstance(filename, str):
            filename = Path(filename)

        filename_unzip = filename.stem if filename.suffix in ['.gz'] \
            else filename
        name = Path(filename_unzip).stem

        protein = Protein(name)
        protein.filename = filename

        return protein


class Chain():
    def __init__(self, protein: Protein, id: str,
                 biopython: Optional[bpChain] = None):
        self.protein = protein
        self.id = id
        self._biopython = biopython

    @property
    def name(self) -> str:
        return f'{self.protein.id}_{self.id}'

    def as_biopython(self) -> bpChain:
        if not self._biopython:
            structure = self.protein.as_biopython()
            chains = [c for c in structure.get_chains() if c.id == self.id]
            self._biopython = chains[0]

        return self._biopython

    def get_residues(self) -> Generator[Residue, None, None]:
        residues = self.as_biopython().get_residues()
        return (Residue(self, r.id, biopython=r) for r in residues)

    def get_sequence(self):
        def _three_to_one(s: str):
            return three_to_one(s) if s in standard_aa_names else '?'

        residues = self.as_biopython().get_residues()
        sequence = [_three_to_one(residue.get_resname())
                    for residue in residues]
        sequence = ''.join(sequence)
        return sequence

    def save(self, filename: Union[str, Path]):
        io = biopdb.PDBIO()
        io.set_structure(self.as_biopython())
        io.save(str(filename))

    def __repr__(self):
        return f'<Chain id={self.id}>'


class Residue():
    def __init__(self, chain: Chain, id: Tuple[str, int, str],
                 biopython: Optional[bpResidue] = None):
        self.chain = chain
        self.id = id
        self._biopython = biopython

    def as_biopython(self) -> bpResidue:
        if not self._biopython:
            chain = self.chain.as_biopython()
            residues = [r for r in chain.get_residues() if r.id == self.id]
            self._biopython = residues[0]

        return self._biopython

    def __repr__(self):
        return f'<Residue id={self.id}>'
