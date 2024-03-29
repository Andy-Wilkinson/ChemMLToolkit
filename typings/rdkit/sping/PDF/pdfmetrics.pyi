"""
This type stub file was generated by pyright.
"""

"""This contains pre-canned text metrics for the PDFgen package, and may also
be used for any other PIDDLE back ends or packages which use the standard
Type 1 postscript fonts.

Its main function is to let you work out the width of strings; it exposes a 
single function, stringwidth(text, fontname), which works out the width of a 
string in the given font. This is an integer defined in em-square units - each
character is defined in a 1000 x 1000 box called the em-square - for a 1-point high
character.  So to convert to points, multiply by 1000 and then by point size.

The AFM loading stuff worked for me but is not being heavily tested, as pre-canning
the widths for the standard 14 fonts in Acrobat Reader is so much more useful. One
could easily extend it to get the exact bounding box for each characterm useful for 
kerning.


The ascent_descent attribute of the module is a dictionary mapping font names
(with the proper Postscript capitalisation) to ascents and descents.  I ought
to sort out the fontname case issue and the resolution of PIDDLE fonts to 
Postscript font names within this module, but have not yet done so.


13th June 1999
"""
StandardEnglishFonts = ...
widths = ...
ascent_descent = ...
def parseAFMfile(filename): # -> List[int]:
  """Returns an array holding the widths of all characters in the font.
    Ultra-crude parser"""
  ...

class FontCache:
  """Loads and caches font width information on demand.  Font names
    converted to lower case for indexing.  Public interface is stringwidth"""
  def __init__(self) -> None:
    ...
  
  def loadfont(self, fontname): # -> None:
    ...
  
  def getfont(self, fontname): # -> list[int]:
    ...
  
  def stringwidth(self, text, font): # -> int:
    ...
  
  def status(self): # -> KeysView[str]:
    ...
  


TheFontCache = ...
stringwidth = ...
