from __future__ import annotations
from typing import Generator, Optional, Union
import gzip
from pathlib import Path
import Bio.PDB as biopdb
from Bio.PDB.Structure import Structure as bpStructure
from Bio.PDB.Chain import Chain as bpChain
from Bio.PDB.Residue import Residue as bpResidue


class Protein():
    def __init__(self):
        self.filename: Optional[Path] = None
        self._biopython: Optional[bpStructure] = None

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
        protein = Protein()

        if isinstance(filename, Path):
            protein.filename = filename
        else:
            protein.filename = Path(filename)

        return protein


class Chain():
    def __init__(self, protein: Protein, id: str,
                 biopython: Optional[bpChain] = None):
        self.protein = protein
        self.id = id
        self._biopython = biopython

    def as_biopython(self) -> bpChain:
        if not self._biopython:
            structure = self.protein.as_biopython()
            chains = [c for c in structure.get_chains() if c.id == self.id]
            self._biopython = chains[0]

        return self._biopython

    def get_residues(self) -> Generator[Residue, None, None]:
        residues = self.as_biopython().get_residues()
        return (Residue(self, r.id, biopython=r) for r in residues)

    def save(self, filename: Union[str, Path]):
        io = biopdb.PDBIO()
        io.set_structure(self.as_biopython())
        io.save(str(filename))

    def __repr__(self):
        return f'<Chain id={self.id}>'


class Residue():
    def __init__(self, chain: Chain, id: str,
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
