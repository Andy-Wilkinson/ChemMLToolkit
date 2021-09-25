"""
This type stub file was generated by pyright.
"""

"""Internal implementation of binana software
(http://nbcr.ucsd.edu/data/sw/hosted/binana/)

"""
class binana_descriptor:
    def __init__(self, protein=...) -> None:
        """ Descriptor build from binana script (as used in NNScore 2.0

        Parameters
        ----------
        protein: oddt.toolkit.Molecule object (default=None)
            Protein object to be used while generating descriptors.
        """
        ...
    
    def set_protein(self, protein): # -> None:
        """ One function to change all relevant proteins

        Parameters
        ----------
        protein: oddt.toolkit.Molecule object
            Protein object to be used while generating descriptors.
            Protein becomes new global and default protein.
        """
        ...
    
    def build(self, ligands, protein=...): # -> Any:
        """ Descriptor building method

        Parameters
        ----------
        ligands: array-like
            An array of generator of oddt.toolkit.Molecule objects for which the descriptor is computed

        protein: oddt.toolkit.Molecule object (default=None)
            Protein object to be used while generating descriptors.
            If none, then the default protein (from constructor) is used.
            Otherwise, protein becomes new global and default protein.

        Returns
        -------
        descs: numpy array, shape=[n_samples, 351]
            An array of binana descriptors, aligned with input ligands
        """
        ...
    
    def __len__(self): # -> Literal[350]:
        """ Returns the dimensions of descriptors """
        ...
    
    def __reduce__(self): # -> tuple[Type[binana_descriptor], tuple[Unknown]]:
        ...
    

