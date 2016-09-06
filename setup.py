#!/usr/bin/env python
from distutils.core import setup

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(name="bananaplots",
      version=find_version("bananas/bananas.py"),
      description="Plot bananas",
      packages=["bananas", "bananas.tests"],
      requires=['scikit_learn', 'matplotlib', 'numpy', 'scipy'],
    )
