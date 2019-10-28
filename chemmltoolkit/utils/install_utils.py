import os
import site
import glob
import shutil
import importlib

from chemmltoolkit.utils.data_utils import get_file, extract_all


def install_conda(package_url):
    """Installs conda packages.

    Warning: This is a hack to allow conda package installation in situations
    when conda is not available (e.g. Google Colab). In general coda should
    be used directly!

    Args:
        package_url: The direct URL to the conda package to install.
    """
    # Download and extract the package

    package_filename = package_url.split('/')[-1]
    package_archive = get_file(package_filename, package_url)
    package_dir = extract_all(package_archive)

    # Copy any library files to the relevant location

    so_files = glob.glob(os.path.join(package_dir, 'lib/*.so.*'))
    for filename in so_files:
        shutil.copy(filename, '/usr/lib/x86_64-linux-gnu/')

    return package_dir


def install_rdkit():
    """Installs RDKit from the conda package.

    This function will download the RDKit conda package, extract the files
    and copy them to the relevant locations. This is useful for situations
    where the conda tool is not available (e.g. Google Colab)
    """
    if importlib.util.find_spec('rdkit'):
        import rdkit
        print(f'RDKit version {rdkit.__version__} already installed')
    else:
        # Download and install the conda packages

        rdkit_url = 'https://anaconda.org/rdkit/rdkit/2019.09.1.0/' + \
            'download/linux-64/rdkit-2019.09.1.0-py37hc20afe1_1.tar.bz2'
        pyboost_url = 'https://anaconda.org/anaconda/py-boost/1.67.0/' + \
            'download/linux-64/py-boost-1.67.0-py37h04863e7_4.tar.bz2'
        libboost_url = 'https://anaconda.org/anaconda/libboost/1.67.0/' + \
            'download/linux-64/libboost-1.67.0-h46d08c1_4.tar.bz2'
        icu_url = 'https://anaconda.org/anaconda/icu/58.2/download/' + \
            'linux-64/icu-58.2-h211956c_0.tar.bz2'

        install_conda(icu_url)
        install_conda(libboost_url)
        install_conda(pyboost_url)
        rdkit_package_dir = install_conda(rdkit_url)

        # Copy additional files to the relevant locations
        # NB: On Colab to '/usr/local/lib/python3.6/dist-packages/rdkit'

        distpackages_dir = site.getsitepackages()[0]

        shutil.copytree(os.path.join(rdkit_package_dir,
                                     'lib/python3.7/site-packages/rdkit'),
                        os.path.join(distpackages_dir, 'rdkit'))

        os.mkdir('/opt/anaconda1anaconda2anaconda3')

        shutil.copytree(os.path.join(rdkit_package_dir, 'share'),
                        '/opt/anaconda1anaconda2anaconda3/share')

        # Check installation success

        import rdkit
        print(f'RDKit version {rdkit.__version__} installed successfully')
