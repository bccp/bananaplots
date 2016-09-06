#!/usr/bin/env python
from distutils.core import setup

setup(name="bananaplots",
      description="Plot bananas",
      packages=["bananaplots", "bananaplots.tests"],
      requires=['scikit-learn'],
    )
