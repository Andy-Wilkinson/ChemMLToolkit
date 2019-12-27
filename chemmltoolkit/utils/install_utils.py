import importlib
import os
import stat
import subprocess
import sys


from chemmltoolkit.utils.data_utils import get_file


def install_conda_package(package_spec,
                          channel=None,
                          conda_location='/usr/local/'):
    """Installs conda packages.

    Note that this will install packages into the current conda environment.
    In many cases this will be the 'base' environment.

    Args:
        package_spec: The package to install.
        channel: The channel to use (or the default channel if None).
        conda_location: The location in which conda is installed.
    """
    conda_args = ['install', '-q', '-y', package_spec]
    if channel:
        conda_args = conda_args + ['-c', channel]

    conda_bin = os.path.join(conda_location, 'condabin/conda')
    subprocess.run([conda_bin] + conda_args)


def install_miniconda(installer_url='https://repo.anaconda.com/miniconda/' +
                                    'Miniconda3-latest-Linux-x86_64.sh',
                      conda_location='/usr/local/'):
    """ Install Miniconda if it does not already exist.

    Args:
        installer_url: The URL to the Miniconda installer.
        conda_location: The location to install Miniconda.
    """
    if os.path.exists(os.path.join(conda_location, 'condabin')):
        import conda
        print(f'Conda version {conda.__version__} already installed')
    else:
        installer_file = get_file('Miniconda3-latest-Linux-x86_64.sh',
                                  installer_url)
        os.chmod(installer_file,
                 os.stat(installer_file).st_mode | stat.S_IEXEC)
        subprocess.run([installer_file, f'-b -f -p {conda_location}'])
        sys.path.append(os.path.join(conda_location,
                                     'lib/python3.7/site-packages/'))
        import conda
        print(f'Conda version {conda.__version__} installed successfully')


def install_rdkit():
    """Installs RDKit.

    This function will install RDKit using conda (installing Miniconda first
    if required)
    """
    if importlib.util.find_spec('rdkit'):
        import rdkit
        print(f'RDKit version {rdkit.__version__} already installed')
    else:
        install_miniconda()
        install_conda_package('rdkit', channel='rdkit')

        # Check installation success

        import rdkit
        print(f'RDKit version {rdkit.__version__} installed successfully')
