import pytest
from chemmltoolkit.features import MoleculeFeaturiser
import chemmltoolkit.features.moleculeFeatures as mf
from rdkit import Chem


class TestMoleculeFeaturiser(object):
    @pytest.mark.parametrize("smiles_input,feature_fns,expected_output", [
        # Tests for individual features
        ('CCO', [mf.logp], [-0.0014000000000000123]),
        ('CCO', [mf.rdkit('MolLogP')], [-0.0014000000000000123]),
        ('CCO', [mf.molwt], [46.069]),
        ('CCO', [mf.rdkit('MolWt')], [46.069]),
        ('CCO', [mf.num_atoms], [3]),
        ('CCO', [mf.num_bonds], [2]),
        ('NCCO', [mf.num_h_donors], [2]),
        ('NCCO', [mf.num_h_acceptors], [2]),
        ('CCO', [mf.num_heavy_atoms], [3]),
        ('CCO', [mf.tpsa], [20.23]),
        # Tests for multiple features
        ('CCO', [mf.molwt, mf.tpsa, mf.num_heavy_atoms], [46.069, 20.23, 3]),
        ('CCO', [mf.rdkit('MolWt'), mf.rdkit('TPSA')], [46.069, 20.23]),
    ])
    def test_process_molecule(self,
                              smiles_input,
                              feature_fns,
                              expected_output):
        featuriser = MoleculeFeaturiser(feature_fns)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)
        assert features == expected_output
        assert featuriser.get_feature_length() == len(expected_output)

    @pytest.mark.parametrize("smiles_input,feature_fns,length,mx,mn", [
        # Tests for individual features
        ('OCCCCO', [mf.fingerprint_atompair(count=False)], 2048, 1, 0),
        ('OCCCCO', [mf.fingerprint_atompair(count=True)], 2048, 5, 0),
        ('OCCCCO', [mf.fingerprint_morgan(radius=2, count=False)], 2048, 1, 0),
        ('OCCCCO', [mf.fingerprint_morgan(radius=2, count=True)], 2048, 4, 0),
        ('OCCCCO', [mf.fingerprint_morgan(radius=3, count=False)], 2048, 1, 0),
        ('OCCCCO', [mf.fingerprint_morgan(radius=3, count=True)], 2048, 4, 0),
    ])
    def test_process_molecule_fingerprint(self,
                                          smiles_input,
                                          feature_fns,
                                          length,
                                          mx,
                                          mn):
        featuriser = MoleculeFeaturiser(feature_fns)
        mol = Chem.MolFromSmiles(smiles_input)
        features = featuriser.process_molecule(mol)

        assert len(features) == length
        assert max(features) == mx
        assert min(features) == mn
        assert featuriser.get_feature_length() == length

    def test_process_all_rdkit(self):
        featuriser = MoleculeFeaturiser(mf.all_rdkit())
        mol = Chem.MolFromSmiles('OCCCCO')
        features = featuriser.process_molecule(mol)
        assert len(features) == 200
        assert features[0] == 8.08625
        assert features[50] == 0.0
        assert features[100] == 0
