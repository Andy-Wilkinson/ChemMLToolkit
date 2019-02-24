import os
import shutil
import urllib3
import certifi
import tarfile
import zipfile

http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())


def get_file(filename,
             url,
             cache_subdir='downloads',
             cache_dir=None):

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.chemmltoolkit')

    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, cache_subdir)
    fpath = os.path.join(datadir, filename)

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if not os.path.exists(fpath):
        print('Downloading data from', url)

        with http.request('GET', url, preload_content=False) as r, \
                open(fpath, 'wb') as out_file:
            shutil.copyfileobj(r, out_file)

    return fpath


def extract_all(archive,
                cache_subdir=None,
                cache_dir=None):

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.chemmltoolkit')

    if cache_subdir is None:
        archive_name = os.path.basename(archive).split(os.path.extsep)[0]
        cache_subdir = os.path.join('downloads', archive_name)

    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, cache_subdir)

    if not os.path.exists(datadir):
        print('Extracting datafile', archive)
        os.makedirs(datadir)

        if archive.endswith(".tar.bz2"):
            with tarfile.open(archive, "r:bz2") as tar:
                tar.extractall(path=datadir)
        elif archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as zfile:
                zfile.extractall(path=datadir)

    return datadir
