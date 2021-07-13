"""
This type stub file was generated by pyright.
"""

from Bio.PDB.AbstractPropertyMap import AbstractResiduePropertyMap

r"""Use the DSSP program to calculate secondary structure and accessibility.

You need to have a working version of DSSP (and a license, free for academic
use) in order to use this. For DSSP, see http://swift.cmbi.ru.nl/gv/dssp/.

The following Accessible surface area (ASA) values can be used, defaulting
to the Sander and Rost values:

    Miller
        Miller et al. 1987 https://doi.org/10.1016/0022-2836(87)90038-6
    Sander
        Sander and Rost 1994 https://doi.org/10.1002/prot.340200303
    Wilke
        Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635

The DSSP codes for secondary structure used here are:

    =====     ====
    Code      Structure
    =====     ====
     H        Alpha helix (4-12)
     B        Isolated beta-bridge residue
     E        Strand
     G        3-10 helix
     I        Pi helix
     T        Turn
     S        Bend
     \-       None
    =====     ====

Usage
-----
The DSSP class can be used to run DSSP on a PDB or mmCIF file, and provides a
handle to the DSSP secondary structure and accessibility.

**Note** that DSSP can only handle one model, and will only run
calculations on the first model in the provided PDB file.

Examples
--------
Typical use::

    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    p = PDBParser()
    structure = p.get_structure("1MOT", "/local-pdb/1mot.pdb")
    model = structure[0]
    dssp = DSSP(model, "/local-pdb/1mot.pdb")

Note that the recent DSSP executable from the DSSP-2 package was
renamed from ``dssp`` to ``mkdssp``. If using a recent DSSP release,
you may need to provide the name of your DSSP executable::

    dssp = DSSP(model, '/local-pdb/1mot.pdb', dssp='mkdssp')

DSSP data is accessed by a tuple - (chain id, residue id)::

    a_key = list(dssp.keys())[2]
    dssp[a_key]

The dssp data returned for a single residue is a tuple in the form:

    ============ ===
    Tuple Index  Value
    ============ ===
    0            DSSP index
    1            Amino acid
    2            Secondary structure
    3            Relative ASA
    4            Phi
    5            Psi
    6            NH-->O_1_relidx
    7            NH-->O_1_energy
    8            O-->NH_1_relidx
    9            O-->NH_1_energy
    10           NH-->O_2_relidx
    11           NH-->O_2_energy
    12           O-->NH_2_relidx
    13           O-->NH_2_energy
    ============ ===

"""
_dssp_cys = ...
residue_max_acc = ...
def ss_to_index(ss): # -> Literal[0, 1, 2] | None:
    """Secondary structure symbol to index.

    H=0
    E=1
    C=2
    """
    ...

def dssp_dict_from_pdb_file(in_file, DSSP=...): # -> tuple[dict[Unknown, Unknown], list[Unknown]]:
    """Create a DSSP dictionary from a PDB file.

    Parameters
    ----------
    in_file : string
        pdb file

    DSSP : string
        DSSP executable (argument to subprocess)

    Returns
    -------
    (out_dict, keys) : tuple
        a dictionary that maps (chainid, resid) to
        amino acid type, secondary structure code and
        accessibility.

    Examples
    --------
    How dssp_dict_frompdb_file could be used::

        from Bio.PDB.DSSP import dssp_dict_from_pdb_file
        dssp_tuple = dssp_dict_from_pdb_file("/local-pdb/1fat.pdb")
        dssp_dict = dssp_tuple[0]
        print(dssp_dict['A',(' ', 1, ' ')])

    """
    ...

def make_dssp_dict(filename): # -> tuple[dict[Unknown, Unknown], list[Unknown]]:
    """DSSP dictionary mapping identifiers to properties.

    Return a DSSP dictionary that maps (chainid, resid) to
    aa, ss and accessibility, from a DSSP file.

    Parameters
    ----------
    filename : string
        the DSSP output file

    """
    ...

class DSSP(AbstractResiduePropertyMap):
    """Run DSSP and parse secondary structure and accessibility.

    Run DSSP on a PDB/mmCIF file, and provide a handle to the
    DSSP secondary structure and accessibility.

    **Note** that DSSP can only handle one model.

    Examples
    --------
    How DSSP could be used::

        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        p = PDBParser()
        structure = p.get_structure("1MOT", "/local-pdb/1mot.pdb")
        model = structure[0]
        dssp = DSSP(model, "/local-pdb/1mot.pdb")
        # DSSP data is accessed by a tuple (chain_id, res_id)
        a_key = list(dssp.keys())[2]
        # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
        # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
        # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
        dssp[a_key]

    """
    def __init__(self, model, in_file, dssp=..., acc_array=..., file_type=...) -> None:
        """Create a DSSP object.

        Parameters
        ----------
        model : Model
            The first model of the structure
        in_file : string
            Either a PDB file or a DSSP file.
        dssp : string
            The dssp executable (ie. the argument to subprocess)
        acc_array : string
            Accessible surface area (ASA) from either Miller et al. (1987),
            Sander & Rost (1994), or Wilke: Tien et al. 2013, as string
            Sander/Wilke/Miller. Defaults to Sander.
        file_type: string
            File type switch: either PDB, MMCIF or DSSP. Inferred from the
            file extension by default.

        """
        ...
    

