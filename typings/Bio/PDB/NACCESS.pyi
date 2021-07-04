"""
This type stub file was generated by pyright.
"""

from Bio.PDB.AbstractPropertyMap import AbstractAtomPropertyMap, AbstractResiduePropertyMap

"""Interface for the program NACCESS.

See: http://wolf.bms.umist.ac.uk/naccess/
Atomic Solvent Accessible Area Calculations

errors likely to occur with the binary:
default values are often due to low default settings in accall.pars
- e.g. max cubes error: change in accall.pars and recompile binary

use naccess -y, naccess -h or naccess -w to include HETATM records
"""
def run_naccess(model, pdb_file, probe_size=..., z_slice=..., naccess=..., temp_path=...): # -> tuple[List[str], List[str]]:
    """Run naccess for a pdb file."""
    ...

def process_rsa_data(rsa_data): # -> dict[Unknown, Unknown]:
    """Process the .rsa output file: residue level SASA data."""
    ...

def process_asa_data(rsa_data): # -> dict[Unknown, Unknown]:
    """Process the .asa output file: atomic level SASA data."""
    ...

class NACCESS(AbstractResiduePropertyMap):
    """Define NACCESS class for residue properties map."""
    def __init__(self, model, pdb_file=..., naccess_binary=..., tmp_directory=...) -> None:
        """Initialize the class."""
        ...
    


class NACCESS_atomic(AbstractAtomPropertyMap):
    """Define NACCESS atomic class for atom properties map."""
    def __init__(self, model, pdb_file=..., naccess_binary=..., tmp_directory=...) -> None:
        """Initialize the class."""
        ...
    


if __name__ == "__main__":
    p = ...
    s = ...
    model = ...
    n = ...
