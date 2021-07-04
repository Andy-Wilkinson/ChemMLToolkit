"""
This type stub file was generated by pyright.
"""

r"""Read and write BGZF compressed files (the GZIP variant used in BAM).

The SAM/BAM file format (Sequence Alignment/Map) comes in a plain text
format (SAM), and a compressed binary format (BAM). The latter uses a
modified form of gzip compression called BGZF (Blocked GNU Zip Format),
which can be applied to any file format to provide compression with
efficient random access. BGZF is described together with the SAM/BAM
file format at http://samtools.sourceforge.net/SAM1.pdf

Please read the text below about 'virtual offsets' before using BGZF
files for random access.

Aim of this module
------------------
The Python gzip library can be used to read BGZF files, since for
decompression they are just (specialised) gzip files. What this
module aims to facilitate is random access to BGZF files (using the
'virtual offset' idea), and writing BGZF files (which means using
suitably sized gzip blocks and writing the extra 'BC' field in the
gzip headers). As in the gzip library, the zlib library is used
internally.

In addition to being required for random access to and writing of
BAM files, the BGZF format can also be used on other sequential
data (in the sense of one record after another), such as most of
the sequence data formats supported in Bio.SeqIO (like FASTA,
FASTQ, GenBank, etc) or large MAF alignments.

The Bio.SeqIO indexing functions use this module to support BGZF files.

Technical Introduction to BGZF
------------------------------
The gzip file format allows multiple compressed blocks, each of which
could be a stand alone gzip file. As an interesting bonus, this means
you can use Unix ``cat`` to combine two or more gzip files into one by
concatenating them. Also, each block can have one of several compression
levels (including uncompressed, which actually takes up a little bit
more space due to the gzip header).

What the BAM designers realised was that while random access to data
stored in traditional gzip files was slow, breaking the file into
gzip blocks would allow fast random access to each block. To access
a particular piece of the decompressed data, you just need to know
which block it starts in (the offset of the gzip block start), and
how far into the (decompressed) contents of the block you need to
read.

One problem with this is finding the gzip block sizes efficiently.
You can do it with a standard gzip file, but it requires every block
to be decompressed -- and that would be rather slow. Additionally
typical gzip files may use very large blocks.

All that differs in BGZF is that compressed size of each gzip block
is limited to 2^16 bytes, and an extra 'BC' field in the gzip header
records this size. Traditional decompression tools can ignore this,
and unzip the file just like any other gzip file.

The point of this is you can look at the first BGZF block, find out
how big it is from this 'BC' header, and thus seek immediately to
the second block, and so on.

The BAM indexing scheme records read positions using a 64 bit
'virtual offset', comprising ``coffset << 16 | uoffset``, where ``coffset``
is the file offset of the BGZF block containing the start of the read
(unsigned integer using up to 64-16 = 48 bits), and ``uoffset`` is the
offset within the (decompressed) block (unsigned 16 bit integer).

This limits you to BAM files where the last block starts by 2^48
bytes, or 256 petabytes, and the decompressed size of each block
is at most 2^16 bytes, or 64kb. Note that this matches the BGZF
'BC' field size which limits the compressed size of each block to
2^16 bytes, allowing for BAM files to use BGZF with no gzip
compression (useful for intermediate files in memory to reduce
CPU load).

Warning about namespaces
------------------------
It is considered a bad idea to use "from XXX import ``*``" in Python, because
it pollutes the namespace. This is a real issue with Bio.bgzf (and the
standard Python library gzip) because they contain a function called open
i.e. Suppose you do this:

>>> from Bio.bgzf import *
>>> print(open.__module__)
Bio.bgzf

Or,

>>> from gzip import *
>>> print(open.__module__)
gzip

Notice that the open function has been replaced. You can "fix" this if you
need to by importing the built-in open function:

>>> from builtins import open

However, what we recommend instead is to use the explicit namespace, e.g.

>>> from Bio import bgzf
>>> print(bgzf.open.__module__)
Bio.bgzf


Examples
--------
This is an ordinary GenBank file compressed using BGZF, so it can
be decompressed using gzip,

>>> import gzip
>>> handle = gzip.open("GenBank/NC_000932.gb.bgz", "r")
>>> assert 0 == handle.tell()
>>> line = handle.readline()
>>> assert 80 == handle.tell()
>>> line = handle.readline()
>>> assert 143 == handle.tell()
>>> data = handle.read(70000)
>>> assert 70143 == handle.tell()
>>> handle.close()

We can also access the file using the BGZF reader - but pay
attention to the file offsets which will be explained below:

>>> handle = BgzfReader("GenBank/NC_000932.gb.bgz", "r")
>>> assert 0 == handle.tell()
>>> print(handle.readline().rstrip())
LOCUS       NC_000932             154478 bp    DNA     circular PLN 15-APR-2009
>>> assert 80 == handle.tell()
>>> print(handle.readline().rstrip())
DEFINITION  Arabidopsis thaliana chloroplast, complete genome.
>>> assert 143 == handle.tell()
>>> data = handle.read(70000)
>>> assert 987828735 == handle.tell()
>>> print(handle.readline().rstrip())
f="GeneID:844718"
>>> print(handle.readline().rstrip())
     CDS             complement(join(84337..84771,85454..85843))
>>> offset = handle.seek(make_virtual_offset(55074, 126))
>>> print(handle.readline().rstrip())
    68521 tatgtcattc gaaattgtat aaagacaact cctatttaat agagctattt gtgcaagtat
>>> handle.close()

Notice the handle's offset looks different as a BGZF file. This
brings us to the key point about BGZF, which is the block structure:

>>> handle = open("GenBank/NC_000932.gb.bgz", "rb")
>>> for values in BgzfBlocks(handle):
...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
Raw start 0, raw length 15073; data start 0, data length 65536
Raw start 15073, raw length 17857; data start 65536, data length 65536
Raw start 32930, raw length 22144; data start 131072, data length 65536
Raw start 55074, raw length 22230; data start 196608, data length 65536
Raw start 77304, raw length 14939; data start 262144, data length 43478
Raw start 92243, raw length 28; data start 305622, data length 0
>>> handle.close()

In this example the first three blocks are 'full' and hold 65536 bytes
of uncompressed data. The fourth block isn't full and holds 43478 bytes.
Finally there is a special empty fifth block which takes 28 bytes on
disk and serves as an 'end of file' (EOF) marker. If this is missing,
it is possible your BGZF file is incomplete.

By reading ahead 70,000 bytes we moved into the second BGZF block,
and at that point the BGZF virtual offsets start to look different
to a simple offset into the decompressed data as exposed by the gzip
library.

As an example, consider seeking to the decompressed position 196734.
Since 196734 = 65536 + 65536 + 65536 + 126 = 65536*3 + 126, this
is equivalent to jumping the first three blocks (which in this
specific example are all size 65536 after decompression - which
does not always hold) and starting at byte 126 of the fourth block
(after decompression). For BGZF, we need to know the fourth block's
offset of 55074 and the offset within the block of 126 to get the
BGZF virtual offset.

>>> print(55074 << 16 | 126)
3609329790
>>> print(bgzf.make_virtual_offset(55074, 126))
3609329790

Thus for this BGZF file, decompressed position 196734 corresponds
to the virtual offset 3609329790. However, another BGZF file with
different contents would have compressed more or less efficiently,
so the compressed blocks would be different sizes. What this means
is the mapping between the uncompressed offset and the compressed
virtual offset depends on the BGZF file you are using.

If you are accessing a BGZF file via this module, just use the
handle.tell() method to note the virtual offset of a position you
may later want to return to using handle.seek().

The catch with BGZF virtual offsets is while they can be compared
(which offset comes first in the file), you cannot safely subtract
them to get the size of the data between them, nor add/subtract
a relative offset.

Of course you can parse this file with Bio.SeqIO using BgzfReader,
although there isn't any benefit over using gzip.open(...), unless
you want to index BGZF compressed sequence files:

>>> from Bio import SeqIO
>>> handle = BgzfReader("GenBank/NC_000932.gb.bgz")
>>> record = SeqIO.read(handle, "genbank")
>>> handle.close()
>>> print(record.id)
NC_000932.1

Text Mode
---------

Like the standard library gzip.open(...), the BGZF code defaults to opening
files in binary mode.

You can request the file be opened in text mode, but beware that this is hard
coded to the simple "latin1" (aka "iso-8859-1") encoding (which includes all
the ASCII characters), which works well with most Western European languages.
However, it is not fully compatible with the more widely used UTF-8 encoding.

In variable width encodings like UTF-8, some single characters in the unicode
text output are represented by multiple bytes in the raw binary form. This is
problematic with BGZF, as we cannot always decode each block in isolation - a
single unicode character could be split over two blocks. This can even happen
with fixed width unicode encodings, as the BGZF block size is not fixed.

Therefore, this module is currently restricted to only support single byte
unicode encodings, such as ASCII, "latin1" (which is a superset of ASCII), or
potentially other character maps (not implemented).

Furthermore, unlike the default text mode on Python 3, we do not attempt to
implement universal new line mode. This transforms the various operating system
new line conventions like Windows (CR LF or "\r\n"), Unix (just LF, "\n"), or
old Macs (just CR, "\r"), into just LF ("\n"). Here we have the same problem -
is "\r" at the end of a block an incomplete Windows style new line?

Instead, you will get the CR ("\r") and LF ("\n") characters as is.

If your data is in UTF-8 or any other incompatible encoding, you must use
binary mode, and decode the appropriate fragments yourself.
"""
_bgzf_magic = ...
_bgzf_header = ...
_bgzf_eof = ...
_bytes_BC = ...
def open(filename, mode=...): # -> BgzfReader | BgzfWriter:
    r"""Open a BGZF file for reading, writing or appending.

    If text mode is requested, in order to avoid multi-byte characters, this is
    hard coded to use the "latin1" encoding, and "\r" and "\n" are passed as is
    (without implementing universal new line mode).

    If your data is in UTF-8 or any other incompatible encoding, you must use
    binary mode, and decode the appropriate fragments yourself.
    """
    ...

def make_virtual_offset(block_start_offset, within_block_offset):
    """Compute a BGZF virtual offset from block start and within block offsets.

    The BAM indexing scheme records read positions using a 64 bit
    'virtual offset', comprising in C terms:

    block_start_offset << 16 | within_block_offset

    Here block_start_offset is the file offset of the BGZF block
    start (unsigned integer using up to 64-16 = 48 bits), and
    within_block_offset within the (decompressed) block (unsigned
    16 bit integer).

    >>> make_virtual_offset(0, 0)
    0
    >>> make_virtual_offset(0, 1)
    1
    >>> make_virtual_offset(0, 2**16 - 1)
    65535
    >>> make_virtual_offset(0, 2**16)
    Traceback (most recent call last):
    ...
    ValueError: Require 0 <= within_block_offset < 2**16, got 65536

    >>> 65536 == make_virtual_offset(1, 0)
    True
    >>> 65537 == make_virtual_offset(1, 1)
    True
    >>> 131071 == make_virtual_offset(1, 2**16 - 1)
    True

    >>> 6553600000 == make_virtual_offset(100000, 0)
    True
    >>> 6553600001 == make_virtual_offset(100000, 1)
    True
    >>> 6553600010 == make_virtual_offset(100000, 10)
    True

    >>> make_virtual_offset(2**48, 0)
    Traceback (most recent call last):
    ...
    ValueError: Require 0 <= block_start_offset < 2**48, got 281474976710656

    """
    ...

def split_virtual_offset(virtual_offset): # -> tuple[Unknown, Unknown]:
    """Divides a 64-bit BGZF virtual offset into block start & within block offsets.

    >>> (100000, 0) == split_virtual_offset(6553600000)
    True
    >>> (100000, 10) == split_virtual_offset(6553600010)
    True

    """
    ...

def BgzfBlocks(handle): # -> Generator[tuple[Unknown, Any, int, int], None, None]:
    """Low level debugging function to inspect BGZF blocks.

    Expects a BGZF compressed file opened in binary read mode using
    the builtin open function. Do not use a handle from this bgzf
    module or the gzip module's open function which will decompress
    the file.

    Returns the block start offset (see virtual offsets), the block
    length (add these for the start of the next block), and the
    decompressed length of the blocks contents (limited to 65536 in
    BGZF), as an iterator - one tuple per BGZF block.

    >>> from builtins import open
    >>> handle = open("SamBam/ex1.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 18239; data start 0, data length 65536
    Raw start 18239, raw length 18223; data start 65536, data length 65536
    Raw start 36462, raw length 18017; data start 131072, data length 65536
    Raw start 54479, raw length 17342; data start 196608, data length 65536
    Raw start 71821, raw length 17715; data start 262144, data length 65536
    Raw start 89536, raw length 17728; data start 327680, data length 65536
    Raw start 107264, raw length 17292; data start 393216, data length 63398
    Raw start 124556, raw length 28; data start 456614, data length 0
    >>> handle.close()

    Indirectly we can tell this file came from an old version of
    samtools because all the blocks (except the final one and the
    dummy empty EOF marker block) are 65536 bytes.  Later versions
    avoid splitting a read between two blocks, and give the header
    its own block (useful to speed up replacing the header). You
    can see this in ex1_refresh.bam created using samtools 0.1.18:

    samtools view -b ex1.bam > ex1_refresh.bam

    >>> handle = open("SamBam/ex1_refresh.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 53; data start 0, data length 38
    Raw start 53, raw length 18195; data start 38, data length 65434
    Raw start 18248, raw length 18190; data start 65472, data length 65409
    Raw start 36438, raw length 18004; data start 130881, data length 65483
    Raw start 54442, raw length 17353; data start 196364, data length 65519
    Raw start 71795, raw length 17708; data start 261883, data length 65411
    Raw start 89503, raw length 17709; data start 327294, data length 65466
    Raw start 107212, raw length 17390; data start 392760, data length 63854
    Raw start 124602, raw length 28; data start 456614, data length 0
    >>> handle.close()

    The above example has no embedded SAM header (thus the first block
    is very small at just 38 bytes of decompressed data), while the next
    example does (a larger block of 103 bytes). Notice that the rest of
    the blocks show the same sizes (they contain the same read data):

    >>> handle = open("SamBam/ex1_header.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 104; data start 0, data length 103
    Raw start 104, raw length 18195; data start 103, data length 65434
    Raw start 18299, raw length 18190; data start 65537, data length 65409
    Raw start 36489, raw length 18004; data start 130946, data length 65483
    Raw start 54493, raw length 17353; data start 196429, data length 65519
    Raw start 71846, raw length 17708; data start 261948, data length 65411
    Raw start 89554, raw length 17709; data start 327359, data length 65466
    Raw start 107263, raw length 17390; data start 392825, data length 63854
    Raw start 124653, raw length 28; data start 456679, data length 0
    >>> handle.close()

    """
    ...

class BgzfReader:
    r"""BGZF reader, acts like a read only handle but seek/tell differ.

    Let's use the BgzfBlocks function to have a peak at the BGZF blocks
    in an example BAM file,

    >>> from builtins import open
    >>> handle = open("SamBam/ex1.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 18239; data start 0, data length 65536
    Raw start 18239, raw length 18223; data start 65536, data length 65536
    Raw start 36462, raw length 18017; data start 131072, data length 65536
    Raw start 54479, raw length 17342; data start 196608, data length 65536
    Raw start 71821, raw length 17715; data start 262144, data length 65536
    Raw start 89536, raw length 17728; data start 327680, data length 65536
    Raw start 107264, raw length 17292; data start 393216, data length 63398
    Raw start 124556, raw length 28; data start 456614, data length 0
    >>> handle.close()

    Now let's see how to use this block information to jump to
    specific parts of the decompressed BAM file:

    >>> handle = BgzfReader("SamBam/ex1.bam", "rb")
    >>> assert 0 == handle.tell()
    >>> magic = handle.read(4)
    >>> assert 4 == handle.tell()

    So far nothing so strange, we got the magic marker used at the
    start of a decompressed BAM file, and the handle position makes
    sense. Now however, let's jump to the end of this block and 4
    bytes into the next block by reading 65536 bytes,

    >>> data = handle.read(65536)
    >>> len(data)
    65536
    >>> assert 1195311108 == handle.tell()

    Expecting 4 + 65536 = 65540 were you? Well this is a BGZF 64-bit
    virtual offset, which means:

    >>> split_virtual_offset(1195311108)
    (18239, 4)

    You should spot 18239 as the start of the second BGZF block, while
    the 4 is the offset into this block. See also make_virtual_offset,

    >>> make_virtual_offset(18239, 4)
    1195311108

    Let's jump back to almost the start of the file,

    >>> make_virtual_offset(0, 2)
    2
    >>> handle.seek(2)
    2
    >>> handle.close()

    Note that you can use the max_cache argument to limit the number of
    BGZF blocks cached in memory. The default is 100, and since each
    block can be up to 64kb, the default cache could take up to 6MB of
    RAM. The cache is not important for reading through the file in one
    pass, but is important for improving performance of random access.
    """
    def __init__(self, filename=..., mode=..., fileobj=..., max_cache=...) -> None:
        """Initialize the class."""
        ...
    
    def tell(self): # -> int:
        """Return a 64-bit unsigned BGZF virtual offset."""
        ...
    
    def seek(self, virtual_offset):
        """Seek to a 64-bit unsigned BGZF virtual offset."""
        ...
    
    def read(self, size=...): # -> Literal['', b'']:
        """Read method for the BGZF module."""
        ...
    
    def readline(self): # -> Literal['', b'']:
        """Read a single line for the BGZF file."""
        ...
    
    def __next__(self):
        """Return the next line."""
        ...
    
    def __iter__(self): # -> BgzfReader:
        """Iterate over the lines in the BGZF file."""
        ...
    
    def close(self): # -> None:
        """Close BGZF file."""
        ...
    
    def seekable(self): # -> Literal[True]:
        """Return True indicating the BGZF supports random access."""
        ...
    
    def isatty(self): # -> Literal[False]:
        """Return True if connected to a TTY device."""
        ...
    
    def fileno(self): # -> int:
        """Return integer file descriptor."""
        ...
    
    def __enter__(self): # -> BgzfReader:
        """Open a file operable with WITH statement."""
        ...
    
    def __exit__(self, type, value, traceback): # -> None:
        """Close a file with WITH statement."""
        ...
    


class BgzfWriter:
    """Define a BGZFWriter object."""
    def __init__(self, filename=..., mode=..., fileobj=..., compresslevel=...) -> None:
        """Initilize the class."""
        ...
    
    def write(self, data): # -> None:
        """Write method for the class."""
        ...
    
    def flush(self): # -> None:
        """Flush data explicitally."""
        ...
    
    def close(self): # -> None:
        """Flush data, write 28 bytes BGZF EOF marker, and close BGZF file.

        samtools will look for a magic EOF marker, just a 28 byte empty BGZF
        block, and if it is missing warns the BAM file may be truncated. In
        addition to samtools writing this block, so too does bgzip - so this
        implementation does too.
        """
        ...
    
    def tell(self): # -> int:
        """Return a BGZF 64-bit virtual offset."""
        ...
    
    def seekable(self): # -> Literal[False]:
        """Return True indicating the BGZF supports random access."""
        ...
    
    def isatty(self): # -> Literal[False]:
        """Return True if connected to a TTY device."""
        ...
    
    def fileno(self): # -> int:
        """Return integer file descriptor."""
        ...
    
    def __enter__(self): # -> BgzfWriter:
        """Open a file operable with WITH statement."""
        ...
    
    def __exit__(self, type, value, traceback): # -> None:
        """Close a file with WITH statement."""
        ...
    


if __name__ == "__main__":
    stdin = ...
    stdout = ...
    w = ...
