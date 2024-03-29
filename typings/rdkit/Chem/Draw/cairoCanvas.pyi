"""
This type stub file was generated by pyright.
"""

import os
from rdkit.Chem.Draw.canvasbase import CanvasBase

have_cairocffi = ...
have_pango = ...
if 'RDK_NOPANGO' not in os.environ:
  ...
if (not hasattr(cairo.ImageSurface, 'get_data') and not hasattr(cairo.ImageSurface, 'get_data_as_rgba')):
  ...
scriptPattern = ...
class Canvas(CanvasBase):
  def __init__(self, image=..., size=..., ctx=..., imageType=..., fileName=...) -> None:
    """
        Canvas can be used in four modes:
        1) using the supplied PIL image
        2) using the supplied cairo context ctx
        3) writing to a file fileName with image type imageType
        4) creating a cairo surface and context within the constructor
        """
    ...
  
  def flush(self): # -> memoryview | None:
    """temporary interface, must be splitted to different methods,
        """
    ...
  
  def addCanvasLine(self, p1, p2, color=..., color2=..., **kwargs): # -> None:
    ...
  
  def addCanvasText(self, text, pos, font, color=..., **kwargs): # -> tuple[Unknown, Unknown, Unknown] | tuple[Unknown | float, Unknown | float, Unknown]:
    ...
  
  def addCanvasPolygon(self, ps, color=..., fill=..., stroke=..., **kwargs): # -> None:
    ...
  
  def addCanvasDashedWedge(self, p1, p2, p3, dash=..., color=..., color2=..., **kwargs): # -> None:
    ...
  
  def addCircle(self, center, radius, color=..., fill=..., stroke=..., alpha=..., **kwargs): # -> None:
    ...
  


