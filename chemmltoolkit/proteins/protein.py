from __future__ import annotations
from typing import Generator, List, Optional, TextIO, Tuple, Union
import gzip
from enum import Flag
from pathlib import Path
import Bio.PDB as biopdb
from Bio.PDB.Structure import Structure as bpStructure
from Bio.PDB.Chain import Chain as bpChain
from Bio.PDB.Residue import Residue as bpResidue
from Bio.PDB.Polypeptide import standard_aa_names
from Bio.PDB.Polypeptide import three_to_one
import oddt


class ResidueType(Flag):
    RESIDUE = 1
    HETERORESIDUE = 2
    WATER = 4
    ALL = RESIDUE | HETERORESIDUE | WATER

    def get_id_str(self):
        return ' ' if ResidueType.RESIDUE in self else '' + \
            'H' if ResidueType.HETERORESIDUE in self else '' + \
            'W' if ResidueType.HETERORESIDUE in self else ''


def _save(entity: Union[Protein, Chain, Residue],
          filename: Union[str, Path, TextIO]) -> None:
    if isinstance(filename, Path):
        filename = str(filename)

    io = biopdb.PDBIO()
    io.set_structure(entity.as_biopython())
    io.save(filename)


class Protein():
    def __init__(self, id: str):
        self.id = id
        self.filename: Optional[Path] = None
        self._biopython: Optional[bpStructure] = None
        self._oddt: Optional[oddt.toolkit.Molecule] = None

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

    def as_oddt(self) -> oddt.toolkit.Molecule:
        if not self._oddt:
            if self.filename:
                mol = next(oddt.toolkit.readfile('pdb', str(self.filename)))
                mol.protein = True

                self._oddt = mol
            else:
                raise Exception('No filename is specified.')

        return self._oddt

    def get_chain(self, chain_id: str) -> Chain:
        return Chain(self, chain_id)

    def get_chains(self) -> Generator[Chain, None, None]:
        chains = self.as_biopython().get_chains()
        return (Chain(self, c.id, biopython=c) for c in chains)

    def save(self, filename: Union[str, Path, TextIO]):
        _save(self, filename)

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
        return f'{self.protein.name}_{self.id}'

    def as_biopython(self) -> bpChain:
        if not self._biopython:
            structure = self.protein.as_biopython()
            chains = [c for c in structure.get_chains() if c.id == self.id]
            self._biopython = chains[0]

        return self._biopython

    def get_residue(self,
                    residue_number: int) -> Optional[Residue]:
        for r in self.as_biopython().get_residues():
            if r.id[1] == residue_number:
                return Residue(self, r.id, biopython=r)
        return None

    def get_residues(self,
                     residue_type: ResidueType = ResidueType.ALL,
                     residue_names: Optional[List[str]] = None,
                     ) -> Generator[Residue, None, None]:
        def filter(r: bpResidue):
            return r.id[0][0] in residue_ids \
                and (r.resname in residue_names if residue_names else True)

        residue_ids = residue_type.get_id_str()
        residues = self.as_biopython().get_residues()
        return (Residue(self, r.id, biopython=r) for r in residues
                if filter(r))

    def get_sequence(self):
        def _three_to_one(s: str):
            return three_to_one(s) if s in standard_aa_names else '?'

        residues = self.as_biopython().get_residues()
        sequence = [_three_to_one(residue.get_resname())
                    for residue in residues]
        sequence = ''.join(sequence)
        return sequence

    def save(self, filename: Union[str, Path, TextIO]):
        _save(self, filename)

    def __repr__(self):
        return f'<Chain id={self.id}>'


class Residue():
    def __init__(self, chain: Chain, id: Tuple[str, int, str],
                 biopython: Optional[bpResidue] = None):
        self.chain = chain
        self.id = id
        self._biopython = biopython

    @property
    def name(self) -> str:
        return f'{self.chain.name}_{self.residue_id}'

    @property
    def num_atoms(self):
        return sum(1 for _ in self.as_biopython().get_atoms())

    @property
    def residue_id(self):
        return f'{self.residue_name}{self.id[1]}'

    @property
    def residue_name(self):
        return self.as_biopython().resname

    def as_biopython(self) -> bpResidue:
        if not self._biopython:
            chain = self.chain.as_biopython()
            residues = [r for r in chain.get_residues() if r.id == self.id]
            self._biopython = residues[0]

        return self._biopython

    def save(self, filename: Union[str, Path, TextIO]):
        _save(self, filename)

    def __repr__(self):
        return f'<Residue id={self.id}>'
