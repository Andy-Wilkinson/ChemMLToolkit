from __future__ import annotations
from typing import Dict
from chemmltoolkit.features.atomFeaturiser import AtomFeaturiser
from chemmltoolkit.features.bondFeaturiser import BondFeaturiser
from rdkit.Chem import Mol
import torch
import dgl


class GraphFeaturiser():
    """Generator for graph-based features.
    Args:
        features: A list of features to generate.
    """

    def __init__(self: GraphFeaturiser,
                 atom_featurisers: Dict[str, AtomFeaturiser],
                 bond_featurisers: Dict[str, BondFeaturiser],
                 add_self_loops: bool = True):
        self.atom_featurisers = atom_featurisers
        self.bond_featurisers = bond_featurisers
        self.add_self_loops = add_self_loops

    def to_dgl(self: GraphFeaturiser, mol: Mol) -> dgl.DGLGraph:
        """Generates a DGL graph from a molecule.

        Args:
            mol: The molecule to featurise.

        Returns:
            A DGL graph of the featurised molecule.
        """
        num_atoms = mol.GetNumAtoms()
        bonds = mol.GetBonds()
        bond_from = [bond.GetBeginAtomIdx() for bond in bonds]
        bond_to = [bond.GetEndAtomIdx() for bond in bonds]

        g = dgl.graph((torch.tensor(bond_from), torch.tensor(
            bond_to)), num_nodes=num_atoms)

        for key, atom_featuriser in self.atom_featurisers.items():
            atom_features = atom_featuriser.process_molecule(mol)
            g.ndata[key] = torch.tensor(atom_features, dtype=torch.float)

        for key, bond_featuriser in self.bond_featurisers.items():
            bond_features = [bond_featuriser.process_bond(
                bond) for bond in bonds]
            g.edata[key] = torch.tensor(bond_features, dtype=torch.float)

        g = dgl.add_reverse_edges(g, copy_edata=True)

        if self.add_self_loops:
            g = dgl.add_self_loop(g)

        return g
