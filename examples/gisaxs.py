# tests/test_data_plotting.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration, data_processing


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_w_si')
START_INDEX = 61514
END_INDEX = 61517
DETX0 = 100.4
INDEX_LIST = [0, 1, 2, 3]
Q_MIN, Q_MAX, Q_NUM = 0, 0.2, 399
QY_FWHM = 0.002

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.calculate_angle(
    DETX0, params_dict_list, image_array)
rho = calibration.calculate_rho(params_dict_list, azimuth_array, image_array)
qx_array, qy_array, qz_array = calibration.calculate_q(
    DETX0, params_dict_list, image_array, rho)
omega = calibration.calculate_omega(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.calibrate_rel_intensity(
    params_dict_list, image_array, omega)

data_plotting.plot_2d_scattering(
    qy_array,
    qz_array,
    image_array_rel,
    index_list=INDEX_LIST)
data_plotting.plot_2d_paralell(
    qx_array,
    qy_array,
    qz_array,
    image_array_rel,
    index_list=INDEX_LIST)

q_1d_oop = np.linspace(Q_MIN, Q_MAX, Q_NUM)
i_1d_oop = data_processing.calculate_1d_oop(
    QY_FWHM,
    Q_MIN,
    Q_MAX,
    Q_NUM,
    qy_array,
    qz_array,
    params_dict_list,
    image_array_rel,
    omega,
    index_list=INDEX_LIST)
data_plotting.plot_1d_compare(
    q_1d_oop,
    i_1d_oop[0],
    q_1d_oop,
    i_1d_oop[3],
    xscale='linear',
    xlabel='qz',
    yunit='a.u.')
