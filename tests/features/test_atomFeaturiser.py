import pytest
from chemmltoolkit.features import AtomFeaturiser
import chemmltoolkit.features.atomFeatures as af
import chemmltoolkit.features.coreFeatures as feat
from rdkit import Chem


class TestAtomFeaturiser(object):
    @pytest.mark.parametrize("smiles_input,feature_fns,expected_output", [
        # We assume RDKit retains atom ordering from SMILEs
        # A couple of test cases here to check this
        ('CCO', [af.atomic_number], [[6], [6], [8]]),
        ('OCC', [af.atomic_number], [[8], [6], [6]]),

        # Tests for individual features
        ('C[NH2+]C', [af.charge], [[0], [1], [0]]),
        ('CCO', [af.degree], [[1], [2], [1]]),
        ('CC=O', [af.degree], [[1], [2], [1]]),
        ('CC=O', [feat.one_hot(af.hybridization)], [
            [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]),
        ('CCO', [af.hydrogens], [[3], [2], [1]]),
        ('CC=O', [af.hydrogens], [[3], [1], [0]]),
        ('CCO', [af.is_aromatic], [[0], [0], [0]]),
        ('c1ccccc1C', [af.is_aromatic], [[1], [1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', [af.is_ring], [[1], [1], [1], [0]]),
        ('c1ccccc1C', [af.is_ring], [[1], [1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', [af.is_ringsize(3)], [[1], [1], [1], [0]]),
        ('C1CC1C', [af.is_ringsize(4)], [[0], [0], [0], [0]]),
        ('C1CCC1C', [af.is_ringsize(4)], [[1], [1], [1], [1], [0]]),
        ('C1CC1C', [af.is_ringsize(5)], [[0], [0], [0], [0]]),
        ('C1CCCC1C', [af.is_ringsize(5)], [[1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', [af.is_ringsize(6)], [[0], [0], [0], [0]]),
        ('C1CCCCC1C', [af.is_ringsize(6)],
            [[1], [1], [1], [1], [1], [1], [0]]),
        ('c1ccccc1C', [af.is_ringsize(6)],
            [[1], [1], [1], [1], [1], [1], [0]]),
        ('C1CC1C', [af.is_ringsize(7)], [[0], [0], [0], [0]]),
        ('C1CCCCCC1C', [af.is_ringsize(7)],
            [[1], [1], [1], [1], [1], [1], [1], [0]]),
        ('[13C]CO', [af.isotope], [[13], [0], [0]]),
        ('[CH2]CO', [af.radical], [[1], [0], [0]]),
        ('CCO', [feat.one_hot(af.symbol, tokens=[' ', 'C', 'O', 'N'])], [
            [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),

        # Tests for multiple features
        ('CCO', [af.atomic_number,
                 feat.one_hot(af.symbol, tokens=[' ', 'C', 'O', 'N']),
                 af.degree], [
            [6, 0, 1, 0, 0, 1],
            [6, 0, 1, 0, 0, 2],
            [8, 0, 0, 1, 0, 1]]),
    ])
    def test_process_molecule(self,
                              smiles_input,
                              feature_fns,
                              expected_output):
        featuriser = AtomFeaturiser(feature_fns)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)
        assert features == expected_output
        assert featuriser.get_feature_length() == len(expected_output[0])
