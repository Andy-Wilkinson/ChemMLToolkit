import pathlib
from setuptools import setup, find_packages
from chemmltoolkit.__version__ import __version__

# The directory containing this file
here = pathlib.Path(__file__).parent

# Read files for inclusion in the setup
readme = (here / "README.md").read_text()
license = (here / "LICENSE").read_text()

# This call to setup() does all the work
setup(
    name="chemmltoolkit",
    version=__version__,
    description="Useful functionality for machine learning in chemistry",
    long_description=readme,
    long_description_content_type="text/markdown",
    license=license,
    url="https://github.com/Andy-Wilkinson/ChemMLToolkit",
    author="Andrew Wilkinson",
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
)
