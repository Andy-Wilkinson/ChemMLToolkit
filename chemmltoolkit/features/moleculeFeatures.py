import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import rdkit.Chem.Descriptors as Desc


def _fingerprint_fn_bits(generator):
    def _fp(mol: Mol):
        fingerprint = generator.GetFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    return _fp


def _fingerprint_fn_count(generator):
    def _fp(mol: Mol):
        fingerprint = generator.GetCountFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    return _fp


def fingerprint_atompair(fpSize=2048, count=False):
    """Atom pair fingerprint (list of int).

    Args:
        fpSize: Size of the generated fingerprint (defaults to 2048).
        count: The default value of False will generate fingerprint bits
            (0 or 1) whereas a value of True will generate the count of each
            fingerprint value.
    """
    generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpSize)

    if count:
        return _fingerprint_fn_count(generator)
    else:
        return _fingerprint_fn_bits(generator)


def fingerprint_morgan(radius, fpSize=2048, count=False):
    """Morgan fingerprint of the specified size (list of int).

    Args:
        radius: The number of iterations to grow the fingerprint.
        fpSize: Size of the generated fingerprint (defaults to 2048).
        count: The default value of False will generate fingerprint bits
            (0 or 1) whereas a value of True will generate the count of each
            fingerprint value.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                          fpSize=fpSize)

    if count:
        return _fingerprint_fn_count(generator)
    else:
        return _fingerprint_fn_bits(generator)


def logp(mol: Mol):
    """Calculated LogP (float)
    """
    return Desc.MolLogP(mol)


def molwt(mol: Mol):
    """Molecular weight (float)
    """
    return Desc.MolWt(mol)


def num_h_donors(mol: Mol):
    """Number of hydrogen bond donors (int)
    """
    return Desc.NumHDonors(mol)


def num_h_acceptors(mol: Mol):
    """Number of hydrogen bond acceptors (int)
    """
    return Desc.NumHAcceptors(mol)


def num_heavy_atoms(mol: Mol):
    """Number of heavy atoms (int)
    """
    return Desc.HeavyAtomCount(mol)


def tpsa(mol: Mol):
    """Total polar surface area (float)
    """
    return Desc.TPSA(mol)
