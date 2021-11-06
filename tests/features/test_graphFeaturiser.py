from rdkit import Chem
from chemmltoolkit.features import AtomFeaturiser
from chemmltoolkit.features import BondFeaturiser
from chemmltoolkit.features import GraphFeaturiser
import chemmltoolkit.features.atomFeatures as af
import chemmltoolkit.features.bondFeatures as bf


class TestGraphFeaturiser(object):
    def test_to_dgl_with_self_loops(self):
        atom_featuriser = AtomFeaturiser([af.atomic_number])
        bond_featuriser = BondFeaturiser([bf.order])
        featuriser = GraphFeaturiser({'xa': atom_featuriser},
                                     {'xb': bond_featuriser},
                                     add_self_loops=True)

        mol = Chem.MolFromSmiles('CCO')
        graph = featuriser.to_dgl(mol)

        assert graph.num_nodes() == 3
        assert graph.num_edges() == 2 * 2 + 3  # 2 bonds + 3 self-loops
        assert 'xa' in graph.ndata
        assert 'xb' in graph.edata
        assert graph.ndata['xa'].tolist() == [[6.], [6.], [8.]]
        assert graph.edata['xb'].tolist() == [[1.], [1.], [1.], [1.],
                                              [0.], [0.], [0.]]

    def test_to_dgl_without_self_loops(self):
        atom_featuriser = AtomFeaturiser([af.atomic_number])
        bond_featuriser = BondFeaturiser([bf.order])
        featuriser = GraphFeaturiser({'xa': atom_featuriser},
                                     {'xb': bond_featuriser},
                                     add_self_loops=False)

        mol = Chem.MolFromSmiles('CCO')
        graph = featuriser.to_dgl(mol)

        assert graph.num_nodes() == 3
        assert graph.num_edges() == 2 * 2  # 2 bonds
        assert 'xa' in graph.ndata
        assert 'xb' in graph.edata
        assert graph.ndata['xa'].tolist() == [[6.], [6.], [8.]]
        assert graph.edata['xb'].tolist() == [[1.], [1.], [1.], [1.]]
