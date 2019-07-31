from chemmltoolkit.utils.list_utils import flatten
import numpy as np
from rdkit.Chem import DataStructs
import rdkit.Chem.Descriptors as Desc
from rdkit.Chem import Mol
from rdkit.Chem import rdFingerprintGenerator


class MoleculeFeaturiser:
    """Generator for a wide range of molecule-based features.

    A number of features are implemented,
        - logp: The calculated LogP (float)
        - molwt: The molecular weight (float)
        - num_h_donors: The number of hydrogen bond donors (int)
        - num_h_acceptors: The number of hydrogen bond acceptors (int)
        - num_heavy_atoms: The number of heavy atoms (int)
        - tpsa: The total polar surface area (float)

    Molecular fingerprints can also be generated,
        - fp_atompair_b: Atom pair fingerprint (bits)
        - fp_atompair_c: Atom pair fingerprint (count)
        - fp_morgan_b2: Morgan fingerprint with radius 2 (bits)
        - fp_morgan_c2: Morgan fingerprint with radius 2 (count)
        - fp_morgan_b3: Morgan fingerprint with radius 3 (bits)
        - fp_morgan_c3: Morgan fingerprint with radius 3 (count)

    Args:
        features: A list of feature name strings to generate.
    """

    def __init__(self,
                 features: list):
        self.features = [self._get_feature(feature) for feature in features]
        self._gen_fp_atompair = None
        self._gen_fp_morgan2 = None
        self._gen_fp_morgan3 = None

    def process_molecule(self, mol: Mol):
        """Generates molecular features.

        Args:
            mol: The molecule to featurise.

        Returns:
            A list of features.
        """

        features = [feature(mol) for feature in self.features]
        return flatten(features)

    def _get_feature(self, name):
        func_name = f'_f_{name}'
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise f'Undefined molecule feature: {name}'

    def _f_logp(self, mol: Mol): return Desc.MolLogP(mol)
    def _f_molwt(self, mol: Mol): return Desc.MolWt(mol)
    def _f_num_h_donors(self, mol: Mol): return Desc.NumHDonors(mol)
    def _f_num_h_acceptors(self, mol: Mol): return Desc.NumHAcceptors(mol)
    def _f_num_heavy_atoms(self, mol: Mol): return Desc.HeavyAtomCount(mol)
    def _f_tpsa(self, mol: Mol): return Desc.TPSA(mol)

    def _f_fp_atompair_b(self, mol: Mol):
        generator = self._get_fp_gen_atompair()
        return self._get_fp_bits(generator, mol)

    def _f_fp_atompair_c(self, mol: Mol):
        generator = self._get_fp_gen_atompair()
        return self._get_fp_count(generator, mol)

    def _f_fp_morgan_b2(self, mol: Mol):
        generator = self._get_fp_gen_morgan_2()
        return self._get_fp_bits(generator, mol)

    def _f_fp_morgan_c2(self, mol: Mol):
        generator = self._get_fp_gen_morgan_2()
        return self._get_fp_count(generator, mol)

    def _f_fp_morgan_b3(self, mol: Mol):
        generator = self._get_fp_gen_morgan_3()
        return self._get_fp_bits(generator, mol)

    def _f_fp_morgan_c3(self, mol: Mol):
        generator = self._get_fp_gen_morgan_3()
        return self._get_fp_count(generator, mol)

    def _get_fp_gen_atompair(self):
        if not self._gen_fp_atompair:
            self._gen_fp_atompair = \
                rdFingerprintGenerator.GetAtomPairGenerator()
        return self._gen_fp_atompair

    def _get_fp_gen_morgan_2(self):
        if not self._gen_fp_morgan2:
            self._gen_fp_morgan2 = \
                rdFingerprintGenerator.GetMorganGenerator(radius=2)
        return self._gen_fp_morgan2

    def _get_fp_gen_morgan_3(self):
        if not self._gen_fp_morgan3:
            self._gen_fp_morgan3 = \
                rdFingerprintGenerator.GetMorganGenerator(radius=3)
        return self._gen_fp_morgan3

    def _get_fp_bits(self, generator, mol: Mol):
        fingerprint = generator.GetFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _get_fp_count(self, generator, mol: Mol):
        fingerprint = generator.GetCountFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
