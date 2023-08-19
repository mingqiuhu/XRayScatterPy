# setup.py
###
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="XRayScatterPy",
    version="1.0",
    author="Mingqiu Hu and Xuchen Gan in Prof. Thomas P. Russell's group",
    author_email="mingqiuhu@mail.pse.umass.edu",
    description="A Python package for processing and analyzing x-ray and neutron scattering and reflectivity data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mingqiuhu/XRayScatterPy",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "tifffile",
        "xmltodict",
        "requests",
        "pyqt5"
        ],
)
