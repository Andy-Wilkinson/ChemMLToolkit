"""
This type stub file was generated by pyright.
"""

from numpy import array, dot, inner, sqrt
from .qcprotmodule import FastCalcRMSDAndRotation

"""Structural alignment using Quaternion Characteristic Polynomial (QCP).

QCPSuperimposer finds the best rotation and translation to put
two point sets on top of each other (minimizing the RMSD). This is
eg. useful to superimpose crystal structures. QCP stands for
Quaternion Characteristic Polynomial, which is used in the algorithm.
"""
class QCPSuperimposer:
    """Quaternion Characteristic Polynomial (QCP) Superimposer.

    QCPSuperimposer finds the best rotation and translation to put
    two point sets on top of each other (minimizing the RMSD). This is
    eg. useful to superimposing 3D structures of proteins.

    QCP stands for Quaternion Characteristic Polynomial, which is used
    in the algorithm.

    Reference:

    Douglas L Theobald (2005), "Rapid calculation of RMSDs using a
    quaternion-based characteristic polynomial.", Acta Crystallogr
    A 61(4):478-480
    """
    def __init__(self) -> None:
        """Initialize the class."""
        ...
    
    def set(self, reference_coords, coords): # -> None:
        """Set the coordinates to be superimposed.

        coords will be put on top of reference_coords.

        - reference_coords: an NxDIM array
        - coords: an NxDIM array

        DIM is the dimension of the points, N is the number
        of points to be superimposed.
        """
        ...
    
    def run(self): # -> None:
        """Superimpose the coordinate sets."""
        ...
    
    def get_transformed(self): # -> Any:
        """Get the transformed coordinate set."""
        ...
    
    def get_rotran(self): # -> tuple[ndarray, Unknown | None]:
        """Right multiplying rotation matrix and translation."""
        ...
    
    def get_init_rms(self): # -> Any:
        """Root mean square deviation of untransformed coordinates."""
        ...
    
    def get_rms(self):
        """Root mean square deviation of superimposed coordinates."""
        ...
    


