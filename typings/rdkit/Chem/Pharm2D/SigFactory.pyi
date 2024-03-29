"""
This type stub file was generated by pyright.
"""

""" contains factory class for producing signatures


"""
_verbose = ...
class SigFactory:
    """

      SigFactory's are used by creating one, setting the relevant
      parameters, then calling the GetSignature() method each time a
      signature is required.

    """
    def __init__(self, featFactory, useCounts=..., minPointCount=..., maxPointCount=..., shortestPathsOnly=..., includeBondOrder=..., skipFeats=..., trianglePruneBins=...) -> None:
        ...
    
    def SetBins(self, bins): # -> None:
        """ bins should be a list of 2-tuples """
        ...
    
    def GetBins(self): # -> None:
        ...
    
    def GetNumBins(self): # -> int:
        ...
    
    def GetSignature(self):
        ...
    
    def GetBitDescriptionAsText(self, bitIdx, includeBins=..., fullPage=...):
        """  returns text with a description of the bit

        **Arguments**

          - bitIdx: an integer bit index

          - includeBins: (optional) if nonzero, information about the bins will be
            included as well

          - fullPage: (optional) if nonzero, html headers and footers will
            be included (so as to make the output a complete page)

        **Returns**

          a string with the HTML

        """
        ...
    
    def GetBitDescription(self, bitIdx): # -> str:
        """  returns a text description of the bit

        **Arguments**

          - bitIdx: an integer bit index

        **Returns**

          a string

        """
        ...
    
    def GetFeatFamilies(self): # -> list[Unknown]:
        ...
    
    def GetMolFeats(self, mol): # -> list[Unknown]:
        ...
    
    def GetBitIdx(self, featIndices, dists, sortIndices=...):
        """ returns the index for a pharmacophore described using a set of
          feature indices and distances

        **Arguments***

          - featIndices: a sequence of feature indices

          - dists: a sequence of distance between the features, only the
            unique distances should be included, and they should be in the
            order defined in Utils.

          - sortIndices : sort the indices

        **Returns**

          the integer bit index

        """
        ...
    
    def GetBitInfo(self, idx): # -> tuple[Unknown, tuple[int, ...], Unknown]:
        """ returns information about the given bit

         **Arguments**

           - idx: the bit index to be considered

         **Returns**

           a 3-tuple:

             1) the number of points in the pharmacophore

             2) the proto-pharmacophore (tuple of pattern indices)

             3) the scaffold (tuple of distance indices)

        """
        ...
    
    def Init(self): # -> None:
        """ Initializes internal parameters.  This **must** be called after
          making any changes to the signature parameters

        """
        ...
    
    def GetSigSize(self): # -> Literal[0]:
        ...
    


