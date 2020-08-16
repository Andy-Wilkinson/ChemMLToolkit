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


def all_rdkit(exception_list=[]):
    """Gets a list of all RDKit descriptors.

    This will return a set of features for all descriptors in the
    rdkit.Chem.Descriptors.descList property. To allow repeatability, this
    list is hardcoded from RDKit 2019.09.3.

    """
    descriptors_2019_09_3 = [
        'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
        'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
        'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
        'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
        'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ',
        'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
        'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
        'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10',
        'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
        'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
        'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2',
        'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8',
        'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
        'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1',
        'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
        'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10',
        'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
        'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
        'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
        'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
        'NumAliphaticRings', 'NumAromaticCarbocycles',
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
        'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
        'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO',
        'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N',
        'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
        'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
        'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole',
        'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
        'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
        'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
        'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine',
        'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide',
        'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss',
        'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
        'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
        'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
        'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
        'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
        'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
        'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
        'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    return [rdkit(descriptor) for descriptor in descriptors_2019_09_3
            if descriptor not in exception_list]


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
        fingerprint_fn = _fingerprint_fn_count(generator)
    else:
        fingerprint_fn = _fingerprint_fn_bits(generator)

    fingerprint_fn.__name__ = 'fingerprint_atompair(' + \
                              f'fpSize={fpSize},count={count})'
    return fingerprint_fn


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
        fingerprint_fn = _fingerprint_fn_count(generator)
    else:
        fingerprint_fn = _fingerprint_fn_bits(generator)

    fingerprint_fn.__name__ = f'fingerprint_morgan(radius={radius},' + \
                              f'fpSize={fpSize},count={count})'
    return fingerprint_fn


def logp(mol: Mol) -> float:
    """Calculated LogP (float).
    """
    return Desc.MolLogP(mol)


def molwt(mol: Mol) -> float:
    """Molecular weight (float).
    """
    return Desc.MolWt(mol)


def num_atoms(mol: Mol) -> int:
    """Total number of atoms (int).
    """
    return mol.GetNumAtoms()


def num_bonds(mol: Mol) -> int:
    """Total number of bonds (int).
    """
    return mol.GetNumBonds()


def num_h_donors(mol: Mol) -> int:
    """Number of hydrogen bond donors (int).
    """
    return Desc.NumHDonors(mol)


def num_h_acceptors(mol: Mol) -> int:
    """Number of hydrogen bond acceptors (int).
    """
    return Desc.NumHAcceptors(mol)


def num_heavy_atoms(mol: Mol) -> int:
    """Number of heavy atoms (int).
    """
    return Desc.HeavyAtomCount(mol)


def rdkit(name: str):
    """Feature from a specified RDKit descriptor.

    The descriptor should be the name for the corresponding descriptor
    in rdkit.Chem.Descriptors.

    Args:
        name: The name of the descriptor.
    """
    descriptor = getattr(Desc, name)

    def _rdkit(mol: Mol):
        return descriptor(mol)
    _rdkit.__name__ = f'rdkit({name})'
    return _rdkit


def tpsa(mol: Mol) -> float:
    """Total polar surface area (float)
    """
    return Desc.TPSA(mol)
