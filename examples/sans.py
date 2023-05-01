# examples/sans.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sans')
FILE_NAME = 'EQSANS_132751_Iqxqy.dat'

data_array = utils.read_sans(DATA_PATH, FILE_NAME, header=3)
data_plotting.plot_2d_columns(data_array[:, 0], data_array[:, 1], data_array[:, 2], shape=(80, 80))