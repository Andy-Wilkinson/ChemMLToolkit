from rdkit import Chem
from chemmltoolkit.processing.conformerGenerator import ConformerGenerator


class TestConformerGenerator(object):
    def test_generate_conformers(self):
        input_mol = Chem.MolFromSmiles('CCCN')

        conformerGenerator = ConformerGenerator()
        mol = conformerGenerator.generate_conformers(input_mol)

        assert input_mol.GetNumConformers() == 0
        assert mol.GetNumConformers() > 0

    def test_generate_conformers_no_forcefield(self):
        input_mol = Chem.MolFromSmiles('CCCN')

        conformerGenerator = ConformerGenerator(force_field=None)
        mol = conformerGenerator.generate_conformers(input_mol)

        assert input_mol.GetNumConformers() == 0
        assert mol.GetNumConformers() > 0
