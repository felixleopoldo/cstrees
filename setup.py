# -*- coding: utf-8 -*-
#
# This file were created by Python Boilerplate. Use boilerplate to start simple
# usable and best-practices compliant Python projects.
#
# Learn more about it at: http://github.com/fabiommendes/python-boilerplate/
#

import os
import codecs
from setuptools import setup, find_packages

# Save version and author to __meta__.py
version = open("VERSION").read().strip()
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "src", "cstrees", "__meta__.py")
meta = (
    """# Automatically created. Please do not edit.
__version__ = '%s'
__author__ = ''
"""
    % version
)
with open(path, "w") as F:
    F.write(meta)


with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = f.readlines()

setup(
    # Basic info
    name="cstrees",
    version=version,
    author="Felix Rios",
    author_email="felix.leopoldo.rios@gmail.com",
    url="https://github.com/felixleopoldo/cstrees",
    description="A Python library for CStrees.",
    # codecs.open('README.rst', 'rb', 'utf8').read(),
    long_description="A Python library for CStrees.",
    # Classifiers (see https://pypi.python.org/pypi?%3Aaction=list_classifiers)
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Software Development :: Libraries",
    ],
    # Packages and dependencies
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[requirements],
    extras_require={
        "dev": [
            "python-boilerplate[dev]",
        ],
    },
    scripts=["scripts/reproduce_uai", "scripts/demo_jss"],
    include_package_data=True,
    # Other configurations
    zip_safe=False,
    platforms="any",
)
