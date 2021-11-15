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
        ('CCO', [af.atomic_mass], [[12.011], [12.011], [15.999]]),
        ('OCC', [af.atomic_mass], [[15.999], [12.011], [12.011]]),
        ('C[NH2+]C', [af.charge], [[0], [1], [0]]),
        ('CCO', [af.charge_gasteiger],
            [[-0.041838486154772384],
             [0.04022058174908642],
             [-0.3966637147812598]]),
        ('CCO', [af.charge_gasteiger_h],
            [[0.07611987477706995],
             [0.11213943784201265],
             [0.21002230656786322]]),
        ('C[C@H](O)(N)', [feat.one_hot(af.chiral_tag)],
            [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        ('C[C@@H](O)(N)', [feat.one_hot(af.chiral_tag)],
            [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        ('CCO', [af.degree], [[1], [2], [1]]),
        ('CC=O', [af.degree], [[1], [2], [1]]),
        ('CC=O', [feat.one_hot(af.hybridization)], [
            [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]),
        ('CCO', [af.hydrogens], [[3], [2], [1]]),
        ('CC=O', [af.hydrogens], [[3], [1], [0]]),
        ('CCO', [af.index], [[0], [1], [2]]),
        ('CCO', [af.is_aromatic], [[0], [0], [0]]),
        ('c1ccccc1C', [af.is_aromatic], [[1], [1], [1], [1], [1], [1], [0]]),
        ('NCCOC', [af.is_hbond_acceptor], [[0], [0], [0], [1], [0]]),
        ('NCCOC', [af.is_hbond_donor], [[1], [0], [0], [0], [0]]),
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
        ('C1CC1[C@H](F)C1CCC1', [af.stereochemistry],
            [[''], [''], [''], ['S'], [''], [''], [''], [''], ['']]),
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
        feature_lengths = featuriser.get_feature_lengths()
        assert features == expected_output
        assert len(feature_lengths) == len(feature_fns)
        assert sum(feature_lengths) == len(expected_output[0])

    @pytest.mark.parametrize("feature_fns,expected_output", [
        # Tests for individual features
        ([af.atomic_number], ['atomic_number']),
        ([af.is_aromatic], ['is_aromatic']),
        ([af.is_ringsize(3)], ['is_ringsize(3)']),
        ([feat.one_hot(af.hybridization)],
            ['one_hot(hybridization, tokens=[SP,SP2,SP3,SP3D,SP3D2], ' + \
                'unknown_token=False)']),
        # Tests for multiple features
        ([af.is_aromatic, af.degree], ['is_aromatic', 'degree']),
    ])
    def test_get_feature_names(self,
                               feature_fns,
                               expected_output):
        featuriser = AtomFeaturiser(feature_fns)
        feature_info = featuriser.get_feature_names()
        assert feature_info == expected_output

    @pytest.mark.parametrize("feature_fns,expected_output", [
        # Tests for multiple features
        ([af.is_aromatic, af.is_ringsize(3), feat.one_hot(af.hybridization),
          af.degree],
            [
                'is_aromatic',
                'is_ringsize(3)',
                'one_hot(hybridization)[SP]',
                'one_hot(hybridization)[SP2]',
                'one_hot(hybridization)[SP3]',
                'one_hot(hybridization)[SP3D]',
                'one_hot(hybridization)[SP3D2]',
                'degree']),
    ])
    def test_get_feature_keys(self,
                              feature_fns,
                              expected_output):
        featuriser = AtomFeaturiser(feature_fns)
        feature_names = featuriser.get_feature_keys()
        assert feature_names == expected_output
