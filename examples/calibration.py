# examples/calibration.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration, data_processing


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_calibration_q')
START_INDEX = 77023
END_INDEX = 77026
DETX0 = 100.4
INDEX_LIST = [0, 1, 2, 3]
Q_MIN, Q_MAX, Q_NUM = 3e-03, 3, 399

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
q_array = np.sqrt(qx_array**2 + qy_array**2 + qz_array**2)
sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.get_rel_intensity(
    params_dict_list, image_array, sr_array)
image_array_abs = calibration.get_abs_intensity(
    params_dict_list, image_array_rel)


header_info, data_array_esaxs, xml_dict = utils.read_grad_file(
    DATA_PATH, '06022022 AgBH ESAXS.grad')
header_info, data_array_saxs, xml_dict = utils.read_grad_file(
    DATA_PATH, '06022022 AgBH SAXS.grad')
header_info, data_array_maxs, xml_dict = utils.read_grad_file(
    DATA_PATH, '06022022 AgBH MAXS.grad')
header_info, data_array_waxs, xml_dict = utils.read_grad_file(
    DATA_PATH, '06022022 AgBH WAXS.grad')


q_1d = np.linspace(Q_MIN, Q_MAX, Q_NUM)
i_1d = data_processing.calculate_1d(
    q_array,
    image_array_abs,
    sr_array,
    q_min=Q_MIN,
    q_max=Q_MAX,
    q_num=Q_NUM,
    index_list=INDEX_LIST)


data_plotting.plot_1d_compare(
    q_1d, i_1d[0], data_array_esaxs[:, 0], data_array_esaxs[:, 1])
data_plotting.plot_1d_compare(
    q_1d, i_1d[1], data_array_saxs[:, 0], data_array_saxs[:, 1])
data_plotting.plot_1d_compare(
    q_1d, i_1d[2], data_array_maxs[:, 0], data_array_maxs[:, 1])
data_plotting.plot_1d_compare(
    q_1d, i_1d[3], data_array_waxs[:, 0], data_array_waxs[:, 1])
