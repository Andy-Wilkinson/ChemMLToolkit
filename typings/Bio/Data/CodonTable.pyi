"""
This type stub file was generated by pyright.
"""

"""Codon tables based on those from the NCBI.

These tables are based on parsing the NCBI file
ftp://ftp.ncbi.nih.gov/entrez/misc/data/gc.prt
using Scripts/update_ncbi_codon_table.py

Last updated at Version 4.4 (May 2019)
"""
unambiguous_dna_by_name = ...
unambiguous_dna_by_id = ...
unambiguous_rna_by_name = ...
unambiguous_rna_by_id = ...
generic_by_name = ...
generic_by_id = ...
ambiguous_dna_by_name = ...
ambiguous_dna_by_id = ...
ambiguous_rna_by_name = ...
ambiguous_rna_by_id = ...
ambiguous_generic_by_name = ...
ambiguous_generic_by_id = ...
standard_dna_table = ...
standard_rna_table = ...
class TranslationError(Exception):
    """Container for translation specific exceptions."""
    ...


class CodonTable:
    """A codon-table, or genetic code."""
    forward_table = ...
    back_table = ...
    start_codons = ...
    stop_codons = ...
    def __init__(self, nucleotide_alphabet=..., protein_alphabet=..., forward_table=..., back_table=..., start_codons=..., stop_codons=...) -> None:
        """Initialize the class."""
        ...
    
    def __str__(self) -> str:
        """Return a simple text representation of the codon table.

        e.g.::

            >>> import Bio.Data.CodonTable
            >>> print(Bio.Data.CodonTable.standard_dna_table)
            Table 1 Standard, SGC0
            <BLANKLINE>
              |  T      |  C      |  A      |  G      |
            --+---------+---------+---------+---------+--
            T | TTT F   | TCT S   | TAT Y   | TGT C   | T
            T | TTC F   | TCC S   | TAC Y   | TGC C   | C
            ...
            G | GTA V   | GCA A   | GAA E   | GGA G   | A
            G | GTG V   | GCG A   | GAG E   | GGG G   | G
            --+---------+---------+---------+---------+--
            >>> print(Bio.Data.CodonTable.generic_by_id[1])
            Table 1 Standard, SGC0
            <BLANKLINE>
              |  U      |  C      |  A      |  G      |
            --+---------+---------+---------+---------+--
            U | UUU F   | UCU S   | UAU Y   | UGU C   | U
            U | UUC F   | UCC S   | UAC Y   | UGC C   | C
            ...
            G | GUA V   | GCA A   | GAA E   | GGA G   | A
            G | GUG V   | GCG A   | GAG E   | GGG G   | G
            --+---------+---------+---------+---------+--
        """
        ...
    


def make_back_table(table, default_stop_codon): # -> dict[Unknown, Unknown]:
    """Back a back-table (naive single codon mapping).

    ONLY RETURNS A SINGLE CODON, chosen from the possible alternatives
    based on their sort order.
    """
    ...

class NCBICodonTable(CodonTable):
    """Codon table for generic nucleotide sequences."""
    nucleotide_alphabet = ...
    protein_alphabet = ...
    def __init__(self, id, names, table, start_codons, stop_codons) -> None:
        """Initialize the class."""
        ...
    
    def __repr__(self): # -> str:
        """Represent the NCBI codon table class as a string for debugging."""
        ...
    


class NCBICodonTableDNA(NCBICodonTable):
    """Codon table for unambiguous DNA sequences."""
    nucleotide_alphabet = ...


class NCBICodonTableRNA(NCBICodonTable):
    """Codon table for unambiguous RNA sequences."""
    nucleotide_alphabet = ...


class AmbiguousCodonTable(CodonTable):
    """Base codon table for ambiguous sequences."""
    def __init__(self, codon_table, ambiguous_nucleotide_alphabet, ambiguous_nucleotide_values, ambiguous_protein_alphabet, ambiguous_protein_values) -> None:
        """Initialize the class."""
        ...
    
    def __getattr__(self, name): # -> Any:
        """Forward attribute lookups to the original table."""
        ...
    


def list_possible_proteins(codon, forward_table, ambiguous_nucleotide_values): # -> list[Unknown]:
    """Return all possible encoded amino acids for ambiguous codon."""
    ...

def list_ambiguous_codons(codons, ambiguous_nucleotide_values):
    """Extend a codon list to include all possible ambigous codons.

    e.g.::

         ['TAG', 'TAA'] -> ['TAG', 'TAA', 'TAR']
         ['UAG', 'UGA'] -> ['UAG', 'UGA', 'URA']

    Note that ['TAG', 'TGA'] -> ['TAG', 'TGA'], this does not add 'TRR'
    (which could also mean 'TAA' or 'TGG').
    Thus only two more codons are added in the following:

    e.g.::

        ['TGA', 'TAA', 'TAG'] -> ['TGA', 'TAA', 'TAG', 'TRA', 'TAR']

    Returns a new (longer) list of codon strings.
    """
    ...

class AmbiguousForwardTable:
    """Forward table for translation of ambiguous nucleotide sequences."""
    def __init__(self, forward_table, ambiguous_nucleotide, ambiguous_protein) -> None:
        """Initialize the class."""
        ...
    
    def __contains__(self, codon): # -> bool:
        """Check if codon works as key for ambiguous forward_table.

        Only returns 'True' if forward_table[codon] returns a value.
        """
        ...
    
    def get(self, codon, failobj=...):
        """Implement get for dictionary-like behaviour."""
        ...
    
    def __getitem__(self, codon):
        """Implement dictionary-like behaviour for AmbiguousForwardTable.

        forward_table[codon] will either return an amino acid letter,
        or throws a KeyError (if codon does not encode an amino acid)
        or a TranslationError (if codon does encode for an amino acid,
        but either is also a stop codon or does encode several amino acids,
        for which no unique letter is available in the given alphabet.
        """
        ...
    


def register_ncbi_table(name, alt_name, id, table, start_codons, stop_codons): # -> None:
    """Turn codon table data into objects (PRIVATE).

    The data is stored in the dictionaries.
    """
    ...

