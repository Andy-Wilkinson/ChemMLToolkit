import pytest
from chemmltoolkit.processing import BondFeaturiser
from rdkit import Chem


class TestBondFeaturiser(object):
    @pytest.mark.parametrize("smiles_input,feature_names,expected_output", [
        # We assume RDKit retains bond ordering from SMILEs
        # A couple of test cases here to check this
        ('CC=O', ['bondtype_onehot'], [
            (0, 1, [1, 0, 0, 0]),
            (1, 2, [0, 1, 0, 0])]),
        ('O=CC', ['bondtype_onehot'], [
            (0, 1, [0, 1, 0, 0]),
            (1, 2, [1, 0, 0, 0])]),

        # Tests for individual features
        ('CC=CC(=O)C', ['is_conjugated'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (3, 5, [0])]),
        ('CC1CC1', ['is_ring'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 1, [1])]),
        ('CC1CC1', ['is_ring3'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 1, [1])]),
        ('CC1CCC1', ['is_ring3'],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 4, [0]),
            (4, 1, [0])]),
        ('CC1CC1', ['is_ring4'],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCC1', ['is_ring4'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 1, [1])]),
        ('CC1CC1', ['is_ring5'],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCCC1', ['is_ring5'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 1, [1])]),
        ('CC1CC1', ['is_ring6'],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCCCC1', ['is_ring6'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 6, [1]), (6, 1, [1])]),
        ('CC1CC1', ['is_ring7'],  [
            (0, 1, [0]), (1, 2, [0]), (2, 3, [0]), (3, 1, [0])]),
        ('CC1CCCCCC1', ['is_ring7'],  [
            (0, 1, [0]), (1, 2, [1]), (2, 3, [1]), (3, 4, [1]),
            (4, 5, [1]), (5, 6, [1]), (6, 7, [1]), (7, 1, [1])]),

        # Tests for multiple features
        ('C1C(=O)C1', ['bondtype_onehot', 'is_ring'], [
            (0, 1, [1, 0, 0, 0, 1]),
            (1, 2, [0, 1, 0, 0, 0]),
            (1, 3, [1, 0, 0, 0, 1]),
            (3, 0, [1, 0, 0, 0, 1])]),
    ])
    def test_process_molecule(self,
                              smiles_input,
                              feature_names,
                              expected_output):
        featuriser = BondFeaturiser(feature_names)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)
        assert features == expected_output
        assert featuriser.get_feature_length() == len(expected_output[0][2])

    @pytest.mark.parametrize("smiles_input,feature_names,expected_output", [
        # We assume RDKit retains atom ordering from SMILEs
        # A couple of test cases here to check this
        ('CC=O', ['bondtype_onehot'], [
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        ('O=CC', ['bondtype_onehot'], [
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),

        # Tests for individual features
        ('CC1CC1', ['is_ring'], [
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]]),
        ('CC1CC1', ['is_ring3'], [
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]]),
        ('CC1CC1', ['is_ring4'], [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]),

        # Tests for multiple features
        ('C1C(=O)C1', ['bondtype_onehot', 'is_ring'], [
            [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]]]),
    ])
    def test_generate_adjacency_matricies(self,
                                          smiles_input,
                                          feature_names,
                                          expected_output):
        featuriser = BondFeaturiser(feature_names)
        mol = Chem.MolFromSmiles(smiles_input)
        adjacency_matricies = featuriser.generate_adjacency_matricies(mol)
        assert (adjacency_matricies == expected_output).all()
