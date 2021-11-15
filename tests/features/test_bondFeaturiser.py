import pytest
from chemmltoolkit.features import BondFeaturiser
import chemmltoolkit.features.bondFeatures as bf
import chemmltoolkit.features.coreFeatures as feat
from rdkit import Chem


class TestBondFeaturiser(object):
    @pytest.mark.parametrize("smiles_input,feature_fns,expected_output", [
        # We assume RDKit retains bond ordering from SMILEs
        # A couple of test cases here to check this
        ('CC=O', [feat.one_hot(bf.bond_type)], [
            (0, 1, [1, 0, 0, 0]),
            (1, 2, [0, 1, 0, 0])]),
        ('O=CC', [feat.one_hot(bf.bond_type)], [
            (0, 1, [0, 1, 0, 0]),
            (1, 2, [1, 0, 0, 0])]),

        # Tests for individual features
        ('CC=O', [bf.index], [
            (0, 1, [0]), (1, 2, [1])]),
        ('CC1CCCCC1', [bf.is_aromatic],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 4, [0]),
            (4, 5, [0]), (5, 6, [0]), (6, 1, [0])]),
        ('Cc1ccccc1', [bf.is_aromatic],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 6, [1]), (6, 1, [1])]),
        ('CC=CC(=O)C', [bf.is_conjugated],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (3, 5, [0])]),
        ('CC1CC1', [bf.is_ring],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 1, [1])]),
        ('CC1CC1', [bf.is_ringsize(3)],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 1, [1])]),
        ('CC1CCC1', [bf.is_ringsize(3)],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 4, [0]),
            (4, 1, [0])]),
        ('CC1CC1', [bf.is_ringsize(4)],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCC1', [bf.is_ringsize(4)],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 1, [1])]),
        ('CC1CC1', [bf.is_ringsize(5)],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCCC1', [bf.is_ringsize(5)],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 1, [1])]),
        ('CC1CC1', [bf.is_ringsize(6)],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCCCC1', [bf.is_ringsize(6)],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 6, [1]), (6, 1, [1])]),
        ('CC1CC1', [bf.is_ringsize(7)],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCCCCC1', [bf.is_ringsize(7)],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 6, [1]), (6, 7, [1]), (7, 1, [1])]),
        ('CC=O', [bf.order], [
            (0, 1, [1.0]), (1, 2, [2.0])]),
        ('Cc1ccccc1', [bf.order],  [
            (0, 1, [1.0]), (1, 2, [1.5]), (2, 3, [1.5]), (3, 4, [1.5]),
            (4, 5, [1.5]), (5, 6, [1.5]), (6, 1, [1.5])]),
        ('CC=CC', [feat.one_hot(bf.stereochemistry)], [
            (0, 1, [1, 0, 0, 0]), (1, 2, [1, 0, 0, 0]), (2, 3, [1, 0, 0, 0])]),
        ('C/C=C/C', [feat.one_hot(bf.stereochemistry)], [
            (0, 1, [1, 0, 0, 0]), (1, 2, [0, 0, 0, 1]), (2, 3, [1, 0, 0, 0])]),
        ('C/C=C\\C', [feat.one_hot(bf.stereochemistry)], [
            (0, 1, [1, 0, 0, 0]), (1, 2, [0, 0, 1, 0]), (2, 3, [1, 0, 0, 0])]),

        # Tests for multiple features
        ('C1C(=O)C1', [feat.one_hot(bf.bond_type), bf.is_ring], [
            (0, 1, [1, 0, 0, 0, 1]),
            (1, 2, [0, 1, 0, 0, 0]),
            (1, 3, [1, 0, 0, 0, 1]),
            (3, 0, [1, 0, 0, 0, 1])]),
    ])
    def test_process_molecule(self,
                              smiles_input,
                              feature_fns,
                              expected_output):
        featuriser = BondFeaturiser(feature_fns)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)
        feature_lengths = featuriser.get_feature_lengths()
        assert features == expected_output
        assert len(feature_lengths) == len(feature_fns)
        assert sum(feature_lengths) == len(expected_output[0][2])

    @pytest.mark.parametrize("smiles_input,feature_fns,expected_output", [
        # We assume RDKit retains atom ordering from SMILEs
        # A couple of test cases here to check this
        ('CC=O', [feat.one_hot(bf.bond_type)], [
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        ('O=CC', [feat.one_hot(bf.bond_type)], [
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),

        # Tests for individual features
        ('CC1CC1', [bf.is_ring], [
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]]),
        ('CC1CC1', [bf.is_ringsize(3)], [
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]]),
        ('CC1CC1', [bf.is_ringsize(4)], [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]),

        # Tests for multiple features
        ('C1C(=O)C1', [feat.one_hot(bf.bond_type), bf.is_ring], [
            [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]]),
    ])
    def test_generate_adjacency_matricies(self,
                                          smiles_input,
                                          feature_fns,
                                          expected_output):
        featuriser = BondFeaturiser(feature_fns)
        mol = Chem.MolFromSmiles(smiles_input)
        adjacency_matricies = featuriser.generate_adjacency_matricies(mol)
        assert (adjacency_matricies == expected_output).all()

    @pytest.mark.parametrize("feature_fns,expected_output", [
        # Tests for individual features
        ([bf.is_aromatic], ['is_aromatic']),
        ([bf.is_ring], ['is_ring']),
        ([bf.is_ringsize(3)], ['is_ringsize(3)']),
        ([feat.one_hot(bf.bond_type)],
            ['one_hot(bond_type, tokens=[SINGLE,DOUBLE,TRIPLE,AROMATIC], ' + \
                'unknown_token=False)']),
        # Tests for multiple features
        ([bf.is_aromatic, bf.is_ring], ['is_aromatic', 'is_ring']),
    ])
    def test_get_feature_names(self,
                               feature_fns,
                               expected_output):
        featuriser = BondFeaturiser(feature_fns)
        feature_info = featuriser.get_feature_names()
        assert feature_info == expected_output

    @pytest.mark.parametrize("feature_fns,expected_output", [
        # Tests for multiple features
        ([bf.is_aromatic, bf.is_ringsize(3), feat.one_hot(bf.bond_type)],
            [
                'is_aromatic',
                'is_ringsize(3)',
                'one_hot(bond_type)[SINGLE]',
                'one_hot(bond_type)[DOUBLE]',
                'one_hot(bond_type)[TRIPLE]',
                'one_hot(bond_type)[AROMATIC]']),
    ])
    def test_get_feature_keys(self,
                              feature_fns,
                              expected_output):
        featuriser = BondFeaturiser(feature_fns)
        feature_names = featuriser.get_feature_keys()
        assert feature_names == expected_output
