import io
import os
from typing import Dict
from setuptools import setup, find_packages

# Package meta-data.
NAME = "ConvAssist"
DESCRIPTION = "A short description of your package"
URL = "https://github.com/yourusername/your-repo"
EMAIL = "your.email@example.com"
AUTHOR = "Your Name"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "1.0.0"

# Read the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about: Dict[str, str] = {}
# project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
if not VERSION:
    with open(os.path.join(here, "ConvAssist", "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# Debug: Print found packages
found_packages = find_packages(where="./", exclude=["tests*", "tests.*"])
print("Found packages:", found_packages)

# Setup function
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=found_packages,
    package_dir={"": "./"},
    include_package_data=True,
    python_requires=REQUIRES_PYTHON,
    install_requires=[
        # List your package dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)