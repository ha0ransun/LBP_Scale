from setuptools import setup
from distutils.command.build import build
from setuptools.command.install import install
from setuptools.command.develop import develop

import os
import subprocess
BASEPATH = os.path.dirname(os.path.abspath(__file__))


setup(name='lbp',
      py_modules=['lbp'],
      install_requires=[
      ]
)