# setup.py
###
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="xray-scatter-py",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for processing and analyzing x-ray scattering, GISAXS, and x-ray reflectivity data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xray-scatter-py",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
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
        "scikit-image",
        "pandas",
        "tifffile",
        "bornagain",
        "xmltodict",
        "requests"
    ],
)
