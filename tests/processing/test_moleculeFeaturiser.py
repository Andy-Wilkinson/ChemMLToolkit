import pytest
from chemmltoolkit.processing import MoleculeFeaturiser
from rdkit import Chem


class TestMoleculeFeaturiser(object):
    @pytest.mark.parametrize("smiles_input,feature_names,expected_output", [
        # Tests for individual features
        ('CCO', ['logp'], [-0.0014000000000000123]),
        ('CCO', ['molwt'], [46.069]),
        ('NCCO', ['num_h_donors'], [2]),
        ('NCCO', ['num_h_acceptors'], [2]),
        ('CCO', ['num_heavy_atoms'], [3]),
        ('CCO', ['tpsa'], [20.23]),
        # Tests for multiple features
        ('CCO', ['molwt', 'tpsa', 'num_heavy_atoms'], [46.069, 20.23, 3]),
    ])
    def test_process_molecule(self,
                              smiles_input,
                              feature_names,
                              expected_output):
        featuriser = MoleculeFeaturiser(feature_names)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)
        assert features == expected_output

    @pytest.mark.parametrize("smiles_input,feature_names,length,mx,mn", [
        # Tests for individual features
        ('OCCCCO', ['fp_atompair_b'], 2048, 1, 0),
        ('OCCCCO', ['fp_atompair_c'], 2048, 5, 0),
        ('OCCCCO', ['fp_morgan_b2'], 2048, 1, 0),
        ('OCCCCO', ['fp_morgan_c2'], 2048, 4, 0),
        ('OCCCCO', ['fp_morgan_b3'], 2048, 1, 0),
        ('OCCCCO', ['fp_morgan_c3'], 2048, 4, 0),

        # Tests for multiple features
        # ('CCO', ['atomic_number', 'symbol_onehot', 'degree'], [
        #     [6, 0, 1, 0, 0, 1],
        #     [6, 0, 1, 0, 0, 2],
        #     [8, 0, 0, 1, 0, 1]]),
    ])
    def test_process_molecule_stats(self,
                                    smiles_input,
                                    feature_names,
                                    length,
                                    mx,
                                    mn):
        featuriser = MoleculeFeaturiser(feature_names)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)

        assert len(features) == length
        assert max(features) == mx
        assert min(features) == mn
