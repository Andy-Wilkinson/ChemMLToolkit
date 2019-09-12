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
        assert featuriser.get_feature_length() == len(expected_output)

    @pytest.mark.parametrize("smiles_input,feature_names,length,mx,mn", [
        # Tests for individual features
        ('OCCCCO', ['fp_atompair_hb'], 2048, 1, 0),
        ('OCCCCO', ['fp_atompair_hc'], 2048, 5, 0),
        ('OCCCCO', ['fp_morgan2_hb'], 2048, 1, 0),
        ('OCCCCO', ['fp_morgan2_hc'], 2048, 4, 0),
        ('OCCCCO', ['fp_morgan3_hb'], 2048, 1, 0),
        ('OCCCCO', ['fp_morgan3_hc'], 2048, 4, 0),
    ])
    def test_process_molecule_fp_hash(self,
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
        assert featuriser.get_feature_length() == length

    @pytest.mark.parametrize("smiles_input,feature_names,length,mx,mn", [
        # Tests for individual features
        ('OCCCCO', ['fp_atompair_sb'], 23, 1, 0),
        ('OCCCCO', ['fp_atompair_sc'], 11, 2, 0),
        ('OCCCCO', ['fp_morgan2_sb'], 25, 1, 0),
        ('OCCCCO', ['fp_morgan2_sc'], 14, 4, 0),
        ('OCCCCO', ['fp_morgan3_sb'], 26, 1, 0),
        ('OCCCCO', ['fp_morgan3_sc'], 15, 4, 0),
    ])
    def test_process_molecule_fp_sparse(self,
                                        smiles_input,
                                        feature_names,
                                        length,
                                        mx,
                                        mn):
        mol = Chem.MolFromSmiles(smiles_input)
        enum_smiles = ['CCO', 'CCN(CC)CC', 'c1ccccc1']
        enum_mols = [Chem.MolFromSmiles(smiles) for smiles in enum_smiles]

        featuriser = MoleculeFeaturiser(feature_names)
        featuriser.enumerate_tokens(enum_mols, feature_names)
        features = featuriser.process_molecule(mol)

        assert len(features) == length
        assert max(features) == mx
        assert min(features) == mn
        assert featuriser.get_feature_length() == length
