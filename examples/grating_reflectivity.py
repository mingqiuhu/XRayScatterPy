# examples/grating_reflectivity.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration, reflectivity


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_grating')
START_INDEX = 79080
END_INDEX = 79280
DETX0 = 100.4
QY_FWHM = 0.1

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
print('obtained params_dict_list, image_array')
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
print('theta_array, azimuth_array')
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
qx_array_gi, qy_array_gi, qz_array_gi = calibration.get_q_gi(
    qx_array, qy_array, qz_array, params_dict_list)
print('obtained qx_array, qy_array, qz_array')

qz_1d, reflectivity_array, spillover_array, total_array = reflectivity.calculate_relative_reflectivity(
    QY_FWHM, 0.01, qy_array_gi, qz_array_gi, params_dict_list, image_array)
normalized_reflectivity, fitted_spillover = reflectivity.calculate_normalized_reflectivity(
    params_dict_list, qz_1d, reflectivity_array, spillover_array, 0.064)

theta_1d = np.degrees(np.arcsin(qz_1d * 1.542 / 4 / np.pi))
data_plotting.plot_1d(qz_1d, normalized_reflectivity, ylabel='reflectivity', xlabel='qz')
data_plotting.plot_1d_compare(
    theta_1d,
    spillover_array,
    theta_1d,
    fitted_spillover,
    xscale='linear',
    yscale='linear',
    xlabel='theta sample',
    ylabel='spillover',
    legend=['experiment', 'fitted'])
data_plotting.plot_1d(
    theta_1d,
    total_array,
    xlabel='theta sample',
    ylabel='total')
np.savetxt('grating.txt', np.stack((qz_1d[:-1], normalized_reflectivity[:-1]), axis=-1))
