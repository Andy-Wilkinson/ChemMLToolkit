import pytest
from chemmltoolkit.processing import AtomFeaturiser
from rdkit import Chem


class TestAtomFeaturiser(object):
    @pytest.mark.parametrize("smiles_input,feature_names,expected_output", [
        # We assume RDKit retains atom ordering from SMILEs
        # A couple of test cases here to check this
        ('CCO', ['atomic_number'], [[6], [6], [8]]),
        ('OCC', ['atomic_number'], [[8], [6], [6]]),

        # Tests for individual features
        ('C[NH2+]C', ['charge'], [[0], [1], [0]]),
        ('CCO', ['degree'], [[1], [2], [1]]),
        ('CC=O', ['degree'], [[1], [2], [1]]),
        ('CC=O', ['hybridisation_onehot'], [
            [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]),
        ('CCO', ['hydrogens'], [[3], [2], [1]]),
        ('CC=O', ['hydrogens'], [[3], [1], [0]]),
        ('CCO', ['is_aromatic'], [[0], [0], [0]]),
        ('c1ccccc1C', ['is_aromatic'], [[1], [1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', ['is_ring'], [[1], [1], [1], [0]]),
        ('c1ccccc1C', ['is_ring'], [[1], [1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', ['is_ring3'], [[1], [1], [1], [0]]),
        ('C1CC1C', ['is_ring4'], [[0], [0], [0], [0]]),
        ('C1CCC1C', ['is_ring4'], [[1], [1], [1], [1], [0]]),
        ('C1CC1C', ['is_ring5'], [[0], [0], [0], [0]]),
        ('C1CCCC1C', ['is_ring5'], [[1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', ['is_ring6'], [[0], [0], [0], [0]]),
        ('C1CCCCC1C', ['is_ring6'], [[1], [1], [1], [1], [1], [1], [0]]),
        ('c1ccccc1C', ['is_ring6'], [[1], [1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', ['is_ring7'], [[0], [0], [0], [0]]),
        ('C1CCCCCC1C', ['is_ring7'], [[1], [1], [1], [1], [1], [1], [1], [0]]),
        ('[13C]CO', ['isotope'], [[13], [0], [0]]),
        ('[CH2]CO', ['radical'], [[1], [0], [0]]),
        ('CCO', ['symbol_onehot'], [
            [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),

        # Tests for multiple features
        ('CCO', ['atomic_number', 'symbol_onehot', 'degree'], [
            [6, 0, 1, 0, 0, 1],
            [6, 0, 1, 0, 0, 2],
            [8, 0, 0, 1, 0, 1]]),
    ])
    def test_process_molecule(self,
                              smiles_input,
                              feature_names,
                              expected_output):
        featuriser = AtomFeaturiser(feature_names,
                                    tokens_symbol=[' ', 'C', 'O', 'N'])
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)
        assert features == expected_output
