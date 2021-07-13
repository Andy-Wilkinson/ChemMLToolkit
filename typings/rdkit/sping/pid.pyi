"""
This type stub file was generated by pyright.
"""

from rdkit.sping.colors import *

"""
PIDDLE (Plug-In Drawing, Does Little Else)
2D Plug-In Drawing System

Magnus Lie Hetland
Andy Robinson
Joseph J. Strout
and others

February-March 1999

On coordinates: units are Big Points, approximately 1/72 inch.
The origin is at the top-left, and coordinates increase down (y)
and to the right (x).

"""
__version_maj_number__ = ...
__version_min_number__ = ...
__version__ = ...
inch = ...
cm = ...
class StateSaver:
    """This is a little utility class for saving and restoring the
          default drawing parameters of a canvas.  To use it, add a line
          like this before changing any of the parameters:

                  saver = StateSaver(myCanvas)

          then, when "saver" goes out of scope, it will automagically
          restore the drawing parameters of myCanvas."""
    def __init__(self, canvas) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    


class Font:
    "This class represents font typeface, size, and style."
    def __init__(self, size=..., bold=..., italic=..., underline=..., face=...) -> None:
        ...
    
    def __cmp__(self, other): # -> Literal[0, 1]:
        """Compare two fonts to see if they're the same."""
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __setattr__(self, name, value): # -> NoReturn:
        ...
    


figureLine = ...
figureArc = ...
figureCurve = ...
keyBksp = ...
keyDel = ...
keyLeft = ...
keyRight = ...
keyUp = ...
keyDown = ...
keyPgUp = ...
keyPgDn = ...
keyHome = ...
keyEnd = ...
keyClear = ...
keyTab = ...
modShift = ...
modControl = ...
class Canvas:
    """This is the base class for a drawing canvas.  The 'plug-in renderers'
          we speak of are really just classes derived from this one, which implement
          the various drawing methods."""
    def __init__(self, size=..., name=...) -> None:
        """Initialize the canvas, and set default drawing parameters.
                    Derived classes should be sure to call this method."""
        ...
    
    def getSize(self): # -> Unknown:
        ...
    
    def isInteractive(self): # -> Literal[0]:
        "Returns 1 if onClick, onOver, and onKey events are possible, 0 otherwise."
        ...
    
    def canUpdate(self): # -> Literal[0]:
        "Returns 1 if the drawing can be meaningfully updated over time \
                    (e.g., screen graphics), 0 otherwise (e.g., drawing to a file)."
        ...
    
    def clear(self): # -> None:
        "Call this to clear and reset the graphics context."
        ...
    
    def flush(self): # -> None:
        "Call this to indicate that any comamnds that have been issued \
                    but which might be buffered should be flushed to the screen"
        ...
    
    def save(self, file=..., format=...): # -> None:
        """For backends that can be save to a file or sent to a
                    stream, create a valid file out of what's currently been
                    drawn on the canvas.  Trigger any finalization here.
                    Though some backends may allow further drawing after this call,
                    presume that this is not possible for maximum portability

                    file may be either a string or a file object with a write method
                         if left as the default, the canvas's current name will be used

                    format may be used to specify the type of file format to use as
                         well as any corresponding extension to use for the filename
                         This is an optional argument and backends may ignore it if
                         they only produce one file format."""
        ...
    
    def setInfoLine(self, s): # -> None:
        "For interactive Canvases, displays the given string in the \
                    'info line' somewhere where the user can probably see it."
        ...
    
    def stringBox(self, s, font=...): # -> tuple[Unknown, Unknown]:
        ...
    
    def stringWidth(self, s, font=...):
        "Return the logical width of the string if it were drawn \
                    in the current font (defaults to self.font)."
        ...
    
    def fontHeight(self, font=...):
        "Find the height of one line of text (baseline to baseline) of the given font."
        ...
    
    def fontAscent(self, font=...):
        "Find the ascent (height above base) of the given font."
        ...
    
    def fontDescent(self, font=...):
        "Find the descent (extent below base) of the given font."
        ...
    
    def arcPoints(self, x1, y1, x2, y2, startAng=..., extent=...): # -> list[Unknown]:
        "Return a list of points approximating the given arc."
        ...
    
    def curvePoints(self, x1, y1, x2, y2, x3, y3, x4, y4): # -> list[tuple[Unknown, Unknown]]:
        "Return a list of points approximating the given Bezier curve."
        ...
    
    def drawMultiLineString(self, s, x, y, font=..., color=..., angle=..., **kwargs): # -> None:
        "Breaks string into lines (on \n, \r, \n\r, or \r\n), and calls drawString on each."
        ...
    
    def drawLine(self, x1, y1, x2, y2, color=..., width=..., dash=..., **kwargs):
        "Draw a straight line between x1,y1 and x2,y2."
        ...
    
    def drawLines(self, lineList, color=..., width=..., dash=..., **kwargs): # -> None:
        "Draw a set of lines of uniform color and width.  \
                    lineList: a list of (x1,y1,x2,y2) line coordinates."
        ...
    
    def drawString(self, s, x, y, font=..., color=..., angle=..., **kwargs):
        "Draw a string starting at location x,y."
        ...
    
    def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, edgeColor=..., edgeWidth=..., fillColor=..., closed=..., dash=..., **kwargs): # -> None:
        "Draw a Bezier curve with control points x1,y1 to x4,y4."
        ...
    
    def drawRect(self, x1, y1, x2, y2, edgeColor=..., edgeWidth=..., fillColor=..., dash=..., **kwargs): # -> None:
        "Draw the rectangle between x1,y1, and x2,y2. \
                    These should have x1<x2 and y1<y2."
        ...
    
    def drawRoundRect(self, x1, y1, x2, y2, rx=..., ry=..., edgeColor=..., edgeWidth=..., fillColor=..., dash=..., **kwargs): # -> None:
        "Draw a rounded rectangle between x1,y1, and x2,y2, \
                    with corners inset as ellipses with x radius rx and y radius ry. \
                    These should have x1<x2, y1<y2, rx>0, and ry>0."
        ...
    
    def drawEllipse(self, x1, y1, x2, y2, edgeColor=..., edgeWidth=..., fillColor=..., dash=..., **kwargs): # -> None:
        "Draw an orthogonal ellipse inscribed within the rectangle x1,y1,x2,y2. \
                    These should have x1<x2 and y1<y2."
        ...
    
    def drawArc(self, x1, y1, x2, y2, startAng=..., extent=..., edgeColor=..., edgeWidth=..., fillColor=..., dash=..., **kwargs): # -> None:
        "Draw a partial ellipse inscribed within the rectangle x1,y1,x2,y2, \
                    starting at startAng degrees and covering extent degrees.   Angles \
                    start with 0 to the right (+x) and increase counter-clockwise. \
                    These should have x1<x2 and y1<y2."
        ...
    
    def drawPolygon(self, pointlist, edgeColor=..., edgeWidth=..., fillColor=..., closed=..., dash=..., **kwargs):
        """drawPolygon(pointlist) -- draws a polygon
                    pointlist: a list of (x,y) tuples defining vertices
                    closed: if 1, adds an extra segment connecting the last point to the first
                    """
        ...
    
    def drawFigure(self, partList, edgeColor=..., edgeWidth=..., fillColor=..., closed=..., dash=..., **kwargs): # -> None:
        """drawFigure(partList) -- draws a complex figure
                    partlist: a set of lines, curves, and arcs defined by a tuple whose
                                      first element is one of figureLine, figureArc, figureCurve
                                      and whose remaining 4, 6, or 8 elements are parameters."""
        ...
    
    def drawImage(self, image, x1, y1, x2=..., y2=..., **kwargs):
        """Draw a PIL Image into the specified rectangle.  If x2 and y2 are
                    omitted, they are calculated from the image size."""
        ...
    


def getFileObject(file, openFlags=...): # -> TextIOWrapper:
    """Common code for every Canvas.save() operation takes a string
          or a potential file object and assures that a valid fileobj is returned"""
    ...

class AffineMatrix:
    def __init__(self, init=...) -> None:
        ...
    
    def scale(self, sx, sy): # -> None:
        ...
    
    def rotate(self, theta): # -> None:
        "counter clockwise rotation in standard SVG/libart coordinate system"
        ...
    
    def translate(self, tx, ty): # -> None:
        ...
    

