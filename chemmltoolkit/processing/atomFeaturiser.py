from chemmltoolkit.utils.list_utils import flatten
from chemmltoolkit.utils.list_utils import one_hot
from rdkit.Chem import Mol
from rdkit.Chem import Atom
from rdkit.Chem import HybridizationType


class AtomFeaturiser:
    """Generator for a wide range of atom-based features.

    A number of features are implemented,
        - atomic_number: The atomic number (int)
        - charge: The formal charge (int)
        - degree: The number of directly bonded neighbours (int)
        - hybridisation_onehot: The hydridisation (one-hot)
        - hydrogens: Total number of hydrogen atoms (int)
        - is_aromatic: If the atom is aromatic (0 or 1)
        - is_ring: If the atom is in a ring (0 or 1)
        - is_ring_3: If the atom is in a 3-membered ring (0 or 1)
        - is_ring_4: If the atom is in a 4-membered ring (0 or 1)
        - is_ring_5: If the atom is in a 5-membered ring (0 or 1)
        - is_ring_6: If the atom is in a 6-membered ring (0 or 1)
        - is_ring_7: If the atom is in a 7-membered ring (0 or 1)
        - isotope: The isotope of the atom (int)
        - radical: Number of radical electrons (int)
        - symbol_onehot: The atomic symbol (one-hot)

    Args:
        features: A list of feature name strings to generate.
        tokens_symbol: List of tokens to use for atom symbols.
            (no default)
        tokens_hydridisation: List of tokens to use for atom hydridisation.
            (defaults to sp, sp2, sp3, sp3d, sp3d2)
    """
    def __init__(self,
                 features: list,
                 tokens_symbol: list = None,
                 tokens_hydridisation: list = [
                     HybridizationType.SP,
                     HybridizationType.SP2,
                     HybridizationType.SP3,
                     HybridizationType.SP3D,
                     HybridizationType.SP3D2
                 ]):
        self.features = [self._get_feature(feature) for feature in features]
        self.tokens_symbol = tokens_symbol
        self.tokens_hydridisation = tokens_hydridisation

    def process_atom(self, atom: Atom):
        """Generates features for an individual atom.

        Args:
            atom: The atom to featurise.

        Returns:
            A list of features.
        """

        features = [feature(atom) for feature in self.features]
        return flatten(features)

    def process_molecule(self, mol: Mol):
        """Generates features for all atoms in a molecule.

        Args:
            mol: The molecule to featurise.

        Returns:
            A nested list of features for all atoms.
        """
        atoms = mol.GetAtoms()
        return [self.process_atom(atom) for atom in atoms]

    def _get_feature(self, name):
        func_name = f'_f_{name}'
        if hasattr(self, func_name):
            return getattr(self, func_name)
        else:
            raise f'Undefined atom feature: {name}'

    def _f_atomic_number(self, atom: Atom): return atom.GetAtomicNum()
    def _f_charge(self, atom: Atom): return atom.GetFormalCharge()
    def _f_degree(self, atom: Atom): return atom.GetDegree()

    def _f_hybridisation_onehot(self, atom: Atom):
        return one_hot(atom.GetHybridization(), self.tokens_hydridisation)

    def _f_hydrogens(self, atom: Atom): return atom.GetTotalNumHs()
    def _f_is_aromatic(self, atom: Atom): return int(atom.GetIsAromatic())
    def _f_is_ring(self, atom: Atom): return int(atom.IsInRing())
    def _f_is_ring3(self, atom: Atom): return int(atom.IsInRingSize(3))
    def _f_is_ring4(self, atom: Atom): return int(atom.IsInRingSize(4))
    def _f_is_ring5(self, atom: Atom): return int(atom.IsInRingSize(5))
    def _f_is_ring6(self, atom: Atom): return int(atom.IsInRingSize(6))
    def _f_is_ring7(self, atom: Atom): return int(atom.IsInRingSize(7))
    def _f_isotope(self, atom: Atom): return atom.GetIsotope()
    def _f_radical(self, atom: Atom): return atom.GetNumRadicalElectrons()

    def _f_symbol_onehot(self, atom: Atom):
        return one_hot(atom.GetSymbol(), self.tokens_symbol)
