# tests/test_data_plotting.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration, gratings


# 78465 - 78665 PHI = -5.5
# 78670 - 78870 PHI = 4.8
# 78875 - 79075 PHI = -0.04
# 79080 - 79280 PHI = 90

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiff_files_grating')

DETX0 = 100.4
INDEX_LIST = [0]

START_INDEX = 79035
PHI = -0.04

END_INDEX = START_INDEX


params_dict_list, image_array = utils.read_multiimage(DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.calculate_angle(DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.calculate_q(DETX0, params_dict_list, image_array)
omega = calibration.calculate_omega(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.calibrate_rel_intensity(params_dict_list, image_array, omega)

data_plotting.plot_2d_scattering(qy_array, qz_array, image_array_rel, index_list=INDEX_LIST)
qx, qy, qz, qx_0, qy_0, qz_0 = gratings.calculate_q(params_dict_list, image_array, phi=PHI)
data_plotting.plot_2d_scattering_withlines(qy_array, qz_array, image_array, qy, qz, index_list=INDEX_LIST)

# data_plotting.plot_3d_grating(qx_array, qy_array, qz_array, image_array, qx, qy, qz, qx_0, qy_0, qz_0, index_list=None, crop=False)
"""
xmin, xmax = np.min(qy_array), np.max(qy_array)
ymin, ymax = np.min(qz_array), np.max(qz_array)

current_index = 0
for PHI in np.linspace(-10, 10, 201):
    qx, qy, qz, qx_0, qy_0, qz_0 = gratings.calculate_q(params_dict_list, image_array, phi=PHI)
    data_plotting.plot_2d_scattering_onlylines(qy, qz, 0.2, PHI, xmin, xmax, ymin, ymax, write=True, current_index=current_index)
    current_index += 1
"""


"""
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiff_files_ps_si')
START_INDEX = 86809
END_INDEX = 87009
DETX0 = 100.4

params_dict_list, image_array = utils.read_multiimage(DATA_PATH, START_INDEX, END_INDEX)
qx_array, qy_array, qz_array = calibration.calculate_q(DETX0, params_dict_list, image_array)
xmin, xmax = np.min(qy_array), np.max(qy_array)
ymin, ymax = np.min(qz_array), np.max(qz_array)
print('obtained params_dict_list, image_array')
qx, qy, qz, qx_0, qy_0, qz_0 = gratings.calculate_q(params_dict_list, image_array, phi=0)
for current_index in range(101, image_array.shape[0]):
    data_plotting.plot_2d_scattering_onlylines(qy[current_index], qz[current_index], float(params_dict_list[current_index]['sample_angle1']), 0, xmin, xmax, ymin, ymax, write=True, current_index=current_index)
"""