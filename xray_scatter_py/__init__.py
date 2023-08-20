# -*- coding: utf-8 -*-
# xray_scatter_py/__init__.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""
This package processes X-ray and neutron scattering and reflectometry data.
"""

from . import calibration  # Calibration of 2D detector images
from . import data_processing  # Calculate 1D integral
from . import data_plotting  # Plotting 2D images and 1D curves
from . import reflectivity  # Calculate reflectivity date from 2D images
from . import gratings  # Calculate reciprocal space of gratings
from . import utils  # Type checking, reading, writing, etc.
from . import nist  # Connect to NIST website to calculate scattering length.
from . import penetration  # Calculate x-ray and neutron penetration depth.
from . import ui  # GUI

__version__ = "1.0"
__author__ = "Mingqiu Hu, Xuchen Gan, Prof. Thomas P. Russell"
__email__ = "mingqiuhu@mail.pse.umass.edu"
__license__ = "MIT"  # Replace with the appropriate license, if needed
