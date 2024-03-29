"""
This type stub file was generated by pyright.
"""

from rdkit.sping.pid import *
from math import *

"""piddleSVG

This module implements an SVG PIDDLE canvas.
In other words, this is a PIDDLE backend that renders into a
SVG file.

Bits have been shamelessly cobbled from piddlePDF.py and/or
piddlePS.py

Greg Landrum (greglandrum@earthlink.net) 3/10/2000
"""
SVG_HEADER = ...
class SVGCanvas(Canvas):
  def __init__(self, size=..., name=..., includeXMLHeader=..., extraHeaderText=...) -> None:
    ...
  
  def clear(self): # -> None:
    ...
  
  def flush(self): # -> None:
    ...
  
  def save(self, file=..., format=...): # -> None:
    """Hand hand this either a file= <filename> or
    file = <an open file object>.  By default, I've made the fomrat extension be
    .svg.  By default it saves the file to "self.name" + '.svg' """
    ...
  
  def text(self): # -> str:
    ...
  
  def drawLine(self, x1, y1, x2, y2, color=..., width=..., dash=..., **kwargs): # -> None:
    "Draw a straight line between x1,y1 and x2,y2."
    ...
  
  def drawPolygon(self, pointlist, edgeColor=..., edgeWidth=..., fillColor=..., closed=..., dash=..., **kwargs): # -> None:
    """drawPolygon(pointlist) -- draws a polygon
    pointlist: a list of (x,y) tuples defining vertices
    """
    ...
  
  def drawEllipse(self, x1, y1, x2, y2, edgeColor=..., edgeWidth=..., fillColor=..., dash=..., **kwargs): # -> None:
    ...
  
  def drawArc(self, x1, y1, x2, y2, theta1=..., extent=..., edgeColor=..., edgeWidth=..., fillColor=..., dash=..., **kwargs): # -> None:
    ...
  
  def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, edgeColor=..., edgeWidth=..., fillColor=..., closed=..., dash=..., **kwargs): # -> None:
    ...
  
  def drawString(self, s, x, y, font=..., color=..., angle=..., **kwargs): # -> None:
    ...
  
  def drawFigure(self, partList, edgeColor=..., edgeWidth=..., fillColor=..., closed=..., dash=..., **kwargs): # -> None:
    """drawFigure(partList) -- draws a complex figure
    partlist: a set of lines, curves, and arcs defined by a tuple whose
    first element is one of figureLine, figureArc, figureCurve
    and whose remaining 4, 6, or 8 elements are parameters."""
    ...
  
  def drawImage(self, image, x1, y1, x2=..., y2=..., **kwargs): # -> None:
    """
      to the best of my knowledge, the only real way to get an image
      into SVG is to read it from a file.  So we'll save out to a PNG
      file, then set a link to that in the SVG.
    """
    ...
  
  def stringWidth(self, s, font=...):
    "Return the logical width of the string if it were drawn \
    in the current font (defaults to self.font)."
    ...
  
  def fontAscent(self, font=...):
    ...
  
  def fontDescent(self, font=...):
    ...
  


def test(): # -> None:
  ...

def dashtest(): # -> None:
  ...

def testit(canvas, s, x, y, font=...): # -> None:
  ...

def test2(): # -> None:
  ...

if __name__ == '__main__':
  ...
