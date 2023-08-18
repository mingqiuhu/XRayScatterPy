# examples/gisaxs.py

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
INDEX_LIST = [3]
Q_MIN, Q_MAX, Q_NUM = 0, 0.2, 399
QY_FWHM = 0.002

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
chi = calibration.get_chi(params_dict_list, azimuth_array, image_array, num_azimuth=90)
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.get_rel_intensity(
    params_dict_list, image_array, sr_array)

qx_array_gi, qy_array_gi, qz_array_gi = calibration.get_q_gi(qx_array, qy_array, qz_array, params_dict_list)

data_plotting.plot_2d(
    qy_array_gi,
    qz_array_gi,
    image_array_rel,
    index_list=INDEX_LIST,
    crop=False)
data_plotting.plot_2d_gi(
    qx_array,
    qy_array,
    qz_array,
    image_array_rel,
    index_list=INDEX_LIST)
data_plotting.plot_2d_gi(
    qx_array_gi,
    qy_array_gi,
    qz_array_gi,
    image_array_rel,
    index_list=INDEX_LIST)
data_plotting.plot_2d_polar(
    azimuth_array,
    qx_array_gi,
    qy_array_gi,
    qz_array_gi,
    image_array_rel,
    params_dict_list,
    index_list=INDEX_LIST)
"""
q_1d_oop = np.linspace(Q_MIN, Q_MAX, Q_NUM)
i_1d_oop = data_processing.calculate_1d_oop(
    qy_array_gi,
    qz_array_gi,
    image_array_rel,
    sr_array,
    qy_fwhm=QY_FWHM,
    qz_min=Q_MIN,
    qz_max=Q_MAX,
    qz_num=Q_NUM,
    index_list=INDEX_LIST)
data_plotting.plot_1d(
    q_1d_oop,
    i_1d_oop[3],
    xscale='linear',
    xlabel='qz',
    ylabel='a.u.')

QPAR_MIN, QPAR_MAX, QPAR_NUM = 0, 0.1, 199
QZ_FWHM = 0.002
qpar_array = np.sqrt(qx_array**2 + qy_array**2)
q_1d_ip = np.linspace(QPAR_MIN, QPAR_MAX, QPAR_NUM)
i_1d_ip = data_processing.calculate_1d_ip(
    qpar_array,
    qz_array_gi,
    image_array_rel,
    sr_array,
    params_dict_list,
    qz_fwhm=QZ_FWHM,
    qpar_min=QPAR_MIN,
    qpar_max=QPAR_MAX,
    qpar_num=QPAR_NUM,
    index_list=INDEX_LIST)
data_plotting.plot_1d(
    q_1d_ip,
    i_1d_ip[3],
    xscale='linear',
    xlabel='q_parallel',
    ylabel='a.u.')
"""
