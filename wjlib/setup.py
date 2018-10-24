import pip
from setuptools import setup, find_packages
from setuptools.command import install

setup(
    name='wjlib',
    version='0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='',
    license='',
    author='wiktor',
    author_email='wiktor.jurasz (on gmail)',
    description='Simple utilities for ML homeworks'
)
