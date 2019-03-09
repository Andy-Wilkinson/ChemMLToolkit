import os
import site
import glob
import shutil
import importlib

from chemmltoolkit.utils.data_utils import get_file, extract_all


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
        # Download and extract the package

        rdkit_package_filename = 'rdkit.tar.bz2'
        rdkit_package_url = 'https://anaconda.org/rdkit/rdkit/2018.09.1.0/' + \
            'download/linux-64/rdkit-2018.09.1.0-py36h71b666b_1.tar.bz2'

        rdkit_package_archive = get_file(
            rdkit_package_filename, rdkit_package_url)
        rdkit_package_dir = extract_all(rdkit_package_archive)

        # Copy the libraries to the relevant locations
        # NB: On Colab to '/usr/local/lib/python3.6/dist-packages/rdkit'

        distpackages_dir = site.getsitepackages()[0]

        shutil.copytree(os.path.join(rdkit_package_dir,
                                     'lib/python3.6/site-packages/rdkit'),
                        os.path.join(distpackages_dir, 'rdkit'))

        so_files = glob.glob(os.path.join(rdkit_package_dir, 'lib/*.so.*'))
        for filename in so_files:
            shutil.copy(filename, '/usr/lib/x86_64-linux-gnu/')

        os.mkdir('/opt/anaconda1anaconda2anaconda3')

        shutil.copytree(os.path.join(rdkit_package_dir, 'share'),
                        '/opt/anaconda1anaconda2anaconda3/share')

        # Create a symbolic link for the libboost_python library as rdkit is
        # compiled with a slightly different naming system in Google Colab

        os.symlink('/usr/lib/x86_64-linux-gnu/libboost_python3-py36.so.1.65.1',
                   '/usr/lib/x86_64-linux-gnu/libboost_python3.so.1.65.1')

        import rdkit
        print(f'RDKit version {rdkit.__version__} installed successfully')
