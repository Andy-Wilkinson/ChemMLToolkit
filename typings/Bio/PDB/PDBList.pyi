"""
This type stub file was generated by pyright.
"""

"""Access the PDB over the internet (e.g. to download structures)."""
class PDBList:
    """Quick access to the structure lists on the PDB or its mirrors.

    This class provides quick access to the structure lists on the
    PDB server or its mirrors. The structure lists contain
    four-letter PDB codes, indicating that structures are
    new, have been modified or are obsolete. The lists are released
    on a weekly basis.

    It also provides a function to retrieve PDB files from the server.
    To use it properly, prepare a directory /pdb or the like,
    where PDB files are stored.

    All available file formats (PDB, PDBx/mmCif, PDBML, mmtf) are supported.
    Please note that large structures (containing >62 chains
    and/or 99999 ATOM lines) are no longer stored as a single PDB file
    and by default (when PDB format selected) are not downloaded.

    Large structures can be downloaded in other formats, including PDBx/mmCif
    or as a .tar file (a collection of PDB-like formatted files for a given
    structure).

    If you want to use this module from inside a proxy, add
    the proxy variable to your environment, e.g. in Unix:
    export HTTP_PROXY='http://realproxy.charite.de:888'
    (This can also be added to ~/.bashrc)
    """
    PDB_REF = ...
    def __init__(self, server=..., pdb=..., obsolete_pdb=..., verbose=...) -> None:
        """Initialize the class with the default server or a custom one.

        Argument pdb is the local path to use, defaulting to the current
        directory at the moment of initialisation.
        """
        ...
    
    @staticmethod
    def get_status_list(url): # -> list[Unknown]:
        """Retrieve a list of pdb codes in the weekly pdb status file from given URL.

        Used by get_recent_changes. Typical contents of the list files parsed
        by this method is now very simply - one PDB name per line.
        """
        ...
    
    def get_recent_changes(self): # -> list[list[Unknown]]:
        """Return three lists of the newest weekly files (added,mod,obsolete).

        Reads the directories with changed entries from the PDB server and
        returns a tuple of three URL's to the files of new, modified and
        obsolete entries from the most recent list. The directory with the
        largest numerical name is used.
        Returns None if something goes wrong.

        Contents of the data/status dir (20031013 would be used);:

            drwxrwxr-x   2 1002     sysadmin     512 Oct  6 18:28 20031006
            drwxrwxr-x   2 1002     sysadmin     512 Oct 14 02:14 20031013
            -rw-r--r--   1 1002     sysadmin    1327 Mar 12  2001 README

        """
        ...
    
    def get_all_entries(self): # -> list[_UrlopenRet]:
        """Retrieve the big file containing all the PDB entries and some annotation.

        Returns a list of PDB codes in the index file.
        """
        ...
    
    def get_all_obsolete(self): # -> list[Unknown]:
        """Return a list of all obsolete entries ever in the PDB.

        Returns a list of all obsolete pdb codes that have ever been
        in the PDB.

        Gets and parses the file from the PDB server in the format
        (the first pdb_code column is the one used). The file looks
        like this::

             LIST OF OBSOLETE COORDINATE ENTRIES AND SUCCESSORS
            OBSLTE    31-JUL-94 116L     216L
            ...
            OBSLTE    29-JAN-96 1HFT     2HFT
            OBSLTE    21-SEP-06 1HFV     2J5X
            OBSLTE    21-NOV-03 1HG6
            OBSLTE    18-JUL-84 1HHB     2HHB 3HHB
            OBSLTE    08-NOV-96 1HID     2HID
            OBSLTE    01-APR-97 1HIU     2HIU
            OBSLTE    14-JAN-04 1HKE     1UUZ
            ...

        """
        ...
    
    def retrieve_pdb_file(self, pdb_code, obsolete=..., pdir=..., file_format=..., overwrite=...): # -> str:
        """Fetch PDB structure file from PDB server, and store it locally.

        The PDB structure's file name is returned as a single string.
        If obsolete ``==`` True, the file will be saved in a special file tree.

        NOTE. The default download format has changed from PDB to PDBx/mmCif

        :param pdb_code: 4-symbols structure Id from PDB (e.g. 3J92).
        :type pdb_code: string

        :param file_format:
            File format. Available options:

            * "mmCif" (default, PDBx/mmCif file),
            * "pdb" (format PDB),
            * "xml" (PDBML/XML format),
            * "mmtf" (highly compressed),
            * "bundle" (PDB formatted archive for large structure}

        :type file_format: string

        :param overwrite: if set to True, existing structure files will be overwritten. Default: False
        :type overwrite: bool

        :param obsolete:
            Has a meaning only for obsolete structures. If True, download the obsolete structure
            to 'obsolete' folder, otherwise download won't be performed.
            This option doesn't work for mmtf format as obsoleted structures aren't stored in mmtf.
            Also doesn't have meaning when parameter pdir is specified.
            Note: make sure that you are about to download the really obsolete structure.
            Trying to download non-obsolete structure into obsolete folder will not work
            and you face the "structure doesn't exists" error.
            Default: False

        :type obsolete: bool

        :param pdir: put the file in this directory (default: create a PDB-style directory tree)
        :type pdir: string

        :return: filename
        :rtype: string
        """
        ...
    
    def update_pdb(self, file_format=...): # -> None:
        """Update your local copy of the PDB files.

        I guess this is the 'most wanted' function from this module.
        It gets the weekly lists of new and modified pdb entries and
        automatically downloads the according PDB files.
        You can call this module as a weekly cron job.
        """
        ...
    
    def download_pdb_files(self, pdb_codes, obsolete=..., pdir=..., file_format=..., overwrite=...): # -> None:
        """Fetch set of PDB structure files from the PDB server and stores them locally.

        The PDB structure's file name is returned as a single string.
        If obsolete ``==`` True, the files will be saved in a special file tree.

        :param pdb_codes: a list of 4-symbols structure Ids from PDB
        :type pdb_codes: list of strings

        :param file_format:
            File format. Available options:

            * "mmCif" (default, PDBx/mmCif file),
            * "pdb" (format PDB),
            * "xml" (PMDML/XML format),
            * "mmtf" (highly compressed),
            * "bundle" (PDB formatted archive for large structure}

        :param overwrite: if set to True, existing structure files will be overwritten. Default: False
        :type overwrite: bool

        :param obsolete:
            Has a meaning only for obsolete structures.
            If True, download the obsolete structure
            to 'obsolete' folder, otherwise download won't be performed.
            This option doesn't work for mmtf format as obsoleted structures are not availbe as mmtf.
            (default: False)

        :type obsolete: bool

        :param pdir: put the file in this directory (default: create a PDB-style directory tree)
        :type pdir: string

        :return: filenames
        :rtype: string
        """
        ...
    
    def download_entire_pdb(self, listfile=..., file_format=...): # -> None:
        """Retrieve all PDB entries not present in the local PDB copy.

        :param listfile: filename to which all PDB codes will be written (optional)

        :param file_format:
            File format. Available options:

            * "mmCif" (default, PDBx/mmCif file),
            * "pdb" (format PDB),
            * "xml" (PMDML/XML format),
            * "mmtf" (highly compressed),
            * "bundle" (PDB formatted archive for large structure}

        NOTE. The default download format has changed from PDB to PDBx/mmCif
        """
        ...
    
    def download_obsolete_entries(self, listfile=..., file_format=...): # -> None:
        """Retrieve all obsolete PDB entries not present in local obsolete PDB copy.

        :param listfile: filename to which all PDB codes will be written (optional)

        :param file_format: file format. Available options:
            "mmCif" (default, PDBx/mmCif file),
            "pdb" (format PDB),
            "xml" (PMDML/XML format),

        NOTE. The default download format has changed from PDB to PDBx/mmCif
        """
        ...
    
    def get_seqres_file(self, savefile=...): # -> None:
        """Retrieve and save a (big) file containing all the sequences of PDB entries."""
        ...
    


if __name__ == "__main__":
    doc = ...
    file_format = ...
    overwrite = ...