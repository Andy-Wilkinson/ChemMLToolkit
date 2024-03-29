"""
This type stub file was generated by pyright.
"""

""" uses pymol to interact with molecules

"""
_server = ...
class MolViewer:
  def __init__(self, host=..., port=..., force=..., **kwargs) -> None:
    ...
  
  def InitializePyMol(self): # -> None:
    """ does some initializations to set up PyMol according to our
    tastes

    """
    ...
  
  def DeleteAll(self): # -> None:
    " blows out everything in the viewer "
    ...
  
  def DeleteAllExcept(self, excludes): # -> None:
    " deletes everything except the items in the provided list of arguments "
    ...
  
  def LoadFile(self, filename, name, showOnly=...): # -> bool | int | float | str | bytes | Tuple[Any, ...] | List[Any] | Dict[Any, Any] | datetime | DateTime | Binary | None:
    """ calls pymol's "load" command on the given filename; the loaded object
    is assigned the name "name"
    """
    ...
  
  def ShowMol(self, mol, name=..., showOnly=..., highlightFeatures=..., molB=..., confId=..., zoom=..., forcePDB=..., showSticks=...): # -> bool | int | float | str | bytes | Tuple[Any, ...] | List[Any] | Dict[Any, Any] | datetime | DateTime | Binary | None:
    """ special case for displaying a molecule or mol block """
    ...
  
  def GetSelectedAtoms(self, whichSelection=...): # -> bool | int | float | str | bytes | Tuple[Any, ...] | List[Any] | Dict[Any, Any] | datetime | DateTime | Binary | None:
    " returns the selected atoms "
    ...
  
  def SelectAtoms(self, itemId, atomIndices, selName=...): # -> None:
    " selects a set of atoms "
    ...
  
  def HighlightAtoms(self, indices, where, extraHighlight=...): # -> None:
    " highlights a set of atoms "
    ...
  
  def SetDisplayStyle(self, obj, style=...): # -> None:
    " change the display style of the specified object "
    ...
  
  def SelectProteinNeighborhood(self, aroundObj, inObj, distance=..., name=..., showSurface=...): # -> None:
    """ selects the area of a protein around a specified object/selection name;
    optionally adds a surface to that """
    ...
  
  def AddPharmacophore(self, locs, colors, label, sphereRad=...): # -> None:
    " adds a set of spheres "
    ...
  
  def SetDisplayUpdate(self, val): # -> None:
    ...
  
  def GetAtomCoords(self, sels): # -> dict[Unknown, Unknown]:
    " returns the coordinates of the selected atoms "
    ...
  
  def HideAll(self): # -> None:
    ...
  
  def HideObject(self, objName): # -> None:
    ...
  
  def DisplayObject(self, objName): # -> None:
    ...
  
  def Redraw(self): # -> None:
    ...
  
  def Zoom(self, objName): # -> None:
    ...
  
  def DisplayHBonds(self, objName, molName, proteinName, molSelText=..., proteinSelText=...): # -> None:
    " toggles display of h bonds between the protein and a specified molecule "
    ...
  
  def DisplayCollisions(self, objName, molName, proteinName, distCutoff=..., color=..., molSelText=..., proteinSelText=...): # -> None:
    " toggles display of collisions between the protein and a specified molecule "
    ...
  
  def SaveFile(self, filename): # -> None:
    ...
  
  def GetPNG(self, h=..., w=..., preDelay=...): # -> Image:
    ...
  


