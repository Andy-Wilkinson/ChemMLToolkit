"""
This type stub file was generated by pyright.
"""

"""Common utility functions for various Bio submodules."""
def find_test_dir(start_dir=...): # -> str:
    """Find the absolute path of Biopython's Tests directory.

    Arguments:
    start_dir -- Initial directory to begin lookup (default to current dir)

    If the directory is not found up the filesystem's root directory, an
    exception will be raised.

    """
    ...

def run_doctest(target_dir=..., *args, **kwargs): # -> None:
    """Run doctest for the importing module."""
    ...

if __name__ == "__main__":
    ...
