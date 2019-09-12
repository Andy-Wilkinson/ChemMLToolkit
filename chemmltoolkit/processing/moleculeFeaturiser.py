from chemmltoolkit.utils.list_utils import flatten
from collections import Counter
import numpy as np
from rdkit.Chem import MolFromSmiles
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
        - fp_atompair_xx: Atom pair fingerprint
        - fp_morgan2_xx: Morgan fingerprint with radius 2
        - fp_morgan3_xx: Morgan fingerprint with radius 3
    Where the suffix (xx) is one of,
        - hb: 1024-bit hash of the fingerprint bits
        - hc: 1024-bit hash of the fingerprint counts
        - sb: Sparse list of fingerprint bits
        - sc: Sparse count of fingerprint counts

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

    def get_feature_length(self) -> int:
        """Calculates the length of the generated feature vector

        Returns:
            The length of the feature vector.
        """
        molecule = MolFromSmiles('CC')
        features = self.process_molecule(molecule)
        return len(features)

    def enumerate_tokens(self,
                         mols: list,
                         features: list,
                         min_count: int = 1):
        """Enumerates a list of molecules extracting all unique tokens.

        This is typically performed on molecules from your training set
        prior to generating features. It is mostly of importance for sparse
        fingerprints, where only features that have at least min_count
        examples different that the rest of the set are used.

        Args:
            mols: The molecules to enumerate for tokens.
            features: A list of feature name strings to generate tokens for.
            min_count: The minimum number of examples that are different
                before a token is included.
        """

        max_count = len(mols) - min_count

        for feature in features:
            func_name = f'_enumerate_{feature}'
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                func(mols, min_count, max_count)

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

    def _f_fp_atompair_hb(self, mol: Mol):
        generator = self._get_fp_gen_atompair()
        return self._get_fp_bits(generator, mol)

    def _f_fp_atompair_hc(self, mol: Mol):
        generator = self._get_fp_gen_atompair()
        return self._get_fp_count(generator, mol)

    def _f_fp_atompair_sb(self, mol: Mol):
        generator = self._get_fp_gen_atompair()
        tokens = self._tokens_fp_atompair_sb
        return self._get_fp_sparse_bits(generator, mol, tokens)

    def _f_fp_atompair_sc(self, mol: Mol):
        generator = self._get_fp_gen_atompair()
        tokens = self._tokens_fp_atompair_sc
        return self._get_fp_sparse_count(generator, mol, tokens)

    def _f_fp_morgan2_hb(self, mol: Mol):
        generator = self._get_fp_gen_morgan2()
        return self._get_fp_bits(generator, mol)

    def _f_fp_morgan2_hc(self, mol: Mol):
        generator = self._get_fp_gen_morgan2()
        return self._get_fp_count(generator, mol)

    def _f_fp_morgan2_sb(self, mol: Mol):
        generator = self._get_fp_gen_morgan2()
        tokens = self._tokens_fp_morgan2_sb
        return self._get_fp_sparse_bits(generator, mol, tokens)

    def _f_fp_morgan2_sc(self, mol: Mol):
        generator = self._get_fp_gen_morgan2()
        tokens = self._tokens_fp_morgan2_sc
        return self._get_fp_sparse_count(generator, mol, tokens)

    def _f_fp_morgan3_hb(self, mol: Mol):
        generator = self._get_fp_gen_morgan3()
        return self._get_fp_bits(generator, mol)

    def _f_fp_morgan3_hc(self, mol: Mol):
        generator = self._get_fp_gen_morgan3()
        return self._get_fp_count(generator, mol)

    def _f_fp_morgan3_sb(self, mol: Mol):
        generator = self._get_fp_gen_morgan3()
        tokens = self._tokens_fp_morgan3_sb
        return self._get_fp_sparse_bits(generator, mol, tokens)

    def _f_fp_morgan3_sc(self, mol: Mol):
        generator = self._get_fp_gen_morgan3()
        tokens = self._tokens_fp_morgan3_sc
        return self._get_fp_sparse_count(generator, mol, tokens)

    def _get_fp_gen_atompair(self):
        if not self._gen_fp_atompair:
            self._gen_fp_atompair = \
                rdFingerprintGenerator.GetAtomPairGenerator()
        return self._gen_fp_atompair

    def _get_fp_gen_morgan2(self):
        if not self._gen_fp_morgan2:
            self._gen_fp_morgan2 = \
                rdFingerprintGenerator.GetMorganGenerator(radius=2)
        return self._gen_fp_morgan2

    def _get_fp_gen_morgan3(self):
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

    def _get_fp_sparse_bits(self, generator, mol: Mol, tokens: list):
        fingerprint = generator.GetSparseFingerprint(mol)
        fp_on_bits = fingerprint.GetOnBits()
        array = [int(bit in fp_on_bits) for bit in tokens]
        return array

    def _get_fp_sparse_count(self, generator, mol: Mol, tokens: list):
        fingerprint = generator.GetSparseCountFingerprint(mol)
        fp_counts = fingerprint.GetNonzeroElements()
        array = [(fp_counts[bit] if bit in fp_counts else 0) for bit in tokens]
        return array

    def _enumerate_fp_atompair_sb(self, mols: list, min: int, max: int):
        generator = self._get_fp_gen_atompair()
        self._tokens_fp_atompair_sb = \
            self._enumerate_fp_sb(generator, mols, min, max)

    def _enumerate_fp_atompair_sc(self, mols: list, min: int, max: int):
        generator = self._get_fp_gen_atompair()
        self._tokens_fp_atompair_sc = \
            self._enumerate_fp_sc(generator, mols, min, max)

    def _enumerate_fp_morgan2_sb(self, mols: list, min: int, max: int):
        generator = self._get_fp_gen_morgan2()
        self._tokens_fp_morgan2_sb = \
            self._enumerate_fp_sb(generator, mols, min, max)

    def _enumerate_fp_morgan2_sc(self, mols: list, min: int, max: int):
        generator = self._get_fp_gen_morgan2()
        self._tokens_fp_morgan2_sc = \
            self._enumerate_fp_sc(generator, mols, min, max)

    def _enumerate_fp_morgan3_sb(self, mols: list, min: int, max: int):
        generator = self._get_fp_gen_morgan3()
        self._tokens_fp_morgan3_sb = \
            self._enumerate_fp_sb(generator, mols, min, max)

    def _enumerate_fp_morgan3_sc(self, mols: list, min: int, max: int):
        generator = self._get_fp_gen_morgan3()
        self._tokens_fp_morgan3_sc = \
            self._enumerate_fp_sc(generator, mols, min, max)

    def _enumerate_fp_sb(self, generator, mols: list, min: int, max: int):
        fp_counts = Counter()

        for mol in mols:
            fp = generator.GetSparseFingerprint(mol)
            fp_counts.update(fp.GetOnBits())

        fp_tokens = [bit for bit, count in fp_counts.items()
                     if count >= min and count <= max]
        return fp_tokens

    def _enumerate_fp_sc(self, generator, mols: list, min: int, max: int):
        fp_counts = Counter()

        for mol in mols:
            fp = generator.GetSparseCountFingerprint(mol)
            fp_counts.update(fp.GetNonzeroElements().keys())

        fp_tokens = [bit for bit, count in fp_counts.items()
                     if count >= min and count <= max]
        return fp_tokens
