from rdkit import Chem
from chemmltoolkit.data.pipelines import MoleculePipeline


class TestMoleculePipeline(object):
    def test_filter(self):
        pipeline = MoleculePipeline.from_list([
            Chem.MolFromSmiles('CCO'),
            Chem.MolFromSmiles('c1ccccc1'),
            Chem.MolFromSmiles('CCOCC'),
        ]).filter(lambda mol: mol.GetNumAtoms() <= 5)

        result_smiles_1 = [Chem.MolToSmiles(mol) for mol in pipeline]

        assert result_smiles_1 == ['CCO', 'CCOCC']

    def test_filter_property(self):
        mol1 = Chem.MolFromSmiles('CCO')
        mol2 = Chem.MolFromSmiles('c1ccccc1')
        mol3 = Chem.MolFromSmiles('CCOCC')

        mol1.SetProp('test_prop', 'X')
        mol2.SetProp('test_prop', 'X')
        mol3.SetProp('test_prop', 'O')

        pipeline = MoleculePipeline.from_list([mol1, mol2, mol3]) \
            .filter_property('test_prop', lambda x: x == 'X')

        result_smiles_1 = [Chem.MolToSmiles(mol) for mol in pipeline]

        assert result_smiles_1 == ['CCO', 'c1ccccc1']

    def test_from_list(self):
        pipeline = MoleculePipeline.from_list([
            Chem.MolFromSmiles('CCO'),
            Chem.MolFromSmiles('c1ccccc1'),
            Chem.MolFromSmiles('CCOCC'),
        ])

        result_smiles_1 = [Chem.MolToSmiles(mol) for mol in pipeline]
        result_smiles_2 = [Chem.MolToSmiles(mol) for mol in pipeline]

        assert result_smiles_1 == ['CCO', 'c1ccccc1', 'CCOCC']
        assert result_smiles_2 == ['CCO', 'c1ccccc1', 'CCOCC']

    def test_from_sdf(self):
        pipeline = MoleculePipeline.from_file_sdf('tests/test_data/simple.sdf')

        result_smiles_1 = [Chem.MolToSmiles(mol) for mol in pipeline]
        result_smiles_2 = [Chem.MolToSmiles(mol) for mol in pipeline]

        assert result_smiles_1 == ['CCO', 'c1ccccc1', 'CCOCC']
        assert result_smiles_2 == ['CCO', 'c1ccccc1', 'CCOCC']
