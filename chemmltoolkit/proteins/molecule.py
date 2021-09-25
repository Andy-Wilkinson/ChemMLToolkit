from pathlib import Path
from typing import Dict, Optional, Union
from . import Residue
from chemmltoolkit.utils.data_utils import get_file
import csv
from io import StringIO
import openbabel.pybel as pybel
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from rdkit.Chem import Mol
import oddt


pdb_ligand_dictionary: Optional[Dict[str, str]] = None


def convert_to_molecule(residue: Residue,
                        template_mol: Optional[Mol] = None,
                        infer_bond_order: bool = True) -> Mol:
    with StringIO() as f:
        residue.save(f)
        residue_pdb_block = f.getvalue()

    mol = None

    # Try to apply bond orders from the PDB ligand entry

    if template_mol:
        pdb_mol = Chem.MolFromPDBBlock(residue_pdb_block)
        try:
            pdb_mol = AssignBondOrdersFromTemplate(template_mol, pdb_mol)
            if pdb_mol.HasSubstructMatch(template_mol):
                mol = pdb_mol
        except ValueError:
            pass

    # Otherwise let OpenBabel determine the bond orders

    if not mol and infer_bond_order:
        residue_mol_block = pybel.readstring(
            'pdb', residue_pdb_block).write('mol')
        mol = Chem.MolFromMolBlock(residue_mol_block)

    # If other attempts fail, return the molecule without bond orders

    if not mol:
        mol = Chem.MolFromPDBBlock(residue_pdb_block)

    return mol


def _get_pdb_ligand_dictionary() -> Dict[str, str]:
    global pdb_ligand_dictionary

    if not pdb_ligand_dictionary:
        url = 'http://ligand-expo.rcsb.org/dictionaries/' + \
            'Components-smiles-stereo-oe.smi'
        file_ligand_dictionary = get_file('Components-smiles-stereo-oe.smi',
                                          url)

        pdb_ligand_dictionary = dict()

        with open(file_ligand_dictionary) as file:
            csv_reader = csv.reader(file, delimiter='\t')
            for row in csv_reader:
                if len(row) > 1:
                    pdb_ligand_dictionary[row[1]] = row[0]

    return pdb_ligand_dictionary


def get_pdb_ligand(residue_name: str) -> Optional[Mol]:
    smiles = _get_pdb_ligand_dictionary().get(residue_name)
    return Chem.MolFromSmiles(smiles) if smiles else None


def read_oddt_molecule(filename: Union[str, Path]) -> oddt.toolkit.Molecule:
    if isinstance(filename, str):
        filename = Path(filename)

    mol = next(oddt.toolkit.readfile('mol', str(filename)))
    return mol
