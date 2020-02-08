"""Setup file for snell_tool"""
from setuptools import setup

setup(
    name='snell',
    version='0.0.1',
    install_requires=['numpy',
                      'scipy',
                      'imageio',
                      'scikit-image', # was scikit-image==0.13.1
                      'matplotlib',
                      'six',
                      'jupyter']
)