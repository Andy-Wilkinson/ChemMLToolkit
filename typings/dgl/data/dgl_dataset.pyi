"""
This type stub file was generated by pyright.
"""

import abc

"""Basic DGL Dataset
"""
class DGLDataset:
    r"""The basic DGL dataset for creating graph datasets.
    This class defines a basic template class for DGL Dataset.
    The following steps will be executed automatically:

      1. Check whether there is a dataset cache on disk
         (already processed and stored on the disk) by
         invoking ``has_cache()``. If true, goto 5.
      2. Call ``download()`` to download the data.
      3. Call ``process()`` to process the data.
      4. Call ``save()`` to save the processed dataset on disk and goto 6.
      5. Call ``load()`` to load the processed dataset from disk.
      6. Done.

    Users can overwite these functions with their
    own data processing logic.

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: same as raw_dir
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
        Default: (), the corresponding hash value is ``'f9065fa7'``.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information

    Attributes
    ----------
    url : str
        The URL to download the dataset
    name : str
        The dataset name
    raw_dir : str
        Raw file directory contains the input data folder
    raw_path : str
        Directory contains the input data files.
        Default : ``os.path.join(self.raw_dir, self.name)``
    save_dir : str
        Directory to save the processed dataset
    save_path : str
        File path to save the processed dataset
    verbose : bool
        Whether to print information
    hash : str
        Hash value for the dataset and the setting.
    """
    def __init__(self, name, url=..., raw_dir=..., save_dir=..., hash_key=..., force_reload=..., verbose=...) -> None:
        ...
    
    def download(self): # -> None:
        r"""Overwite to realize your own logic of downloading data.

        It is recommended to download the to the :obj:`self.raw_dir`
        folder. Can be ignored if the dataset is
        already in :obj:`self.raw_dir`.
        """
        ...
    
    def save(self): # -> None:
        r"""Overwite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``dgl.data.utils.save_graphs``
        to save dgl graph into files and use
        ``dgl.data.utils.save_info`` to save extra
        information into files.
        """
        ...
    
    def load(self): # -> None:
        r"""Overwite to realize your own logic of
        loading the saved dataset from files.

        It is recommended to use ``dgl.data.utils.load_graphs``
        to load dgl graph from files and use
        ``dgl.data.utils.load_info`` to load extra information
        into python dict object.
        """
        ...
    
    def process(self):
        r"""Overwrite to realize your own logic of processing the input data.
        """
        ...
    
    def has_cache(self): # -> Literal[False]:
        r"""Overwrite to realize your own logic of
        deciding whether there exists a cached dataset.

        By default False.
        """
        ...
    
    @property
    def url(self):
        r"""Get url to download the raw dataset.
        """
        ...
    
    @property
    def name(self):
        r"""Name of the dataset.
        """
        ...
    
    @property
    def raw_dir(self): # -> str:
        r"""Raw file directory contains the input data folder.
        """
        ...
    
    @property
    def raw_path(self): # -> str:
        r"""Directory contains the input data files.
            By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        ...
    
    @property
    def save_dir(self): # -> str:
        r"""Directory to save the processed dataset.
        """
        ...
    
    @property
    def save_path(self): # -> str:
        r"""Path to save the processed dataset.
        """
        ...
    
    @property
    def verbose(self):
        r"""Whether to print information.
        """
        ...
    
    @property
    def hash(self): # -> str:
        r"""Hash value for the dataset and the setting.
        """
        ...
    
    @abc.abstractmethod
    def __getitem__(self, idx): # -> None:
        r"""Gets the data object at index.
        """
        ...
    
    @abc.abstractmethod
    def __len__(self): # -> None:
        r"""The number of examples in the dataset."""
        ...
    


class DGLBuiltinDataset(DGLDataset):
    r"""The Basic DGL Builtin Dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    url : str
        Url to download the raw dataset.
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: False
    """
    def __init__(self, name, url, raw_dir=..., hash_key=..., force_reload=..., verbose=...) -> None:
        ...
    
    def download(self): # -> None:
        r""" Automatically download data and extract it.
        """
        ...
    


