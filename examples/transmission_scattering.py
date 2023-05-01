# examples/transmission_scattering.py

import os
from xray_scatter_py import data_plotting, utils, calibration


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_calibration_q')
START_INDEX = 77023
END_INDEX = 77026
DETX0 = 100.4
INDEX_LIST = [0, 1, 2, 3]

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.get_rel_intensity(
    params_dict_list, image_array, sr_array)
image_array_abs = calibration.get_abs_intensity(
    params_dict_list, image_array_rel)
data_plotting.plot_2d(
    qy_array,
    qz_array,
    image_array_abs,
    index_list=INDEX_LIST)
data_plotting.plot_2d_polar(
    azimuth_array,
    qx_array,
    qy_array,
    qz_array,
    image_array_abs,
    index_list=INDEX_LIST)

data_plotting.plot_3d_q(
    qx_array,
    qy_array,
    qz_array,
    image_array_abs,
    index_list=INDEX_LIST)

x_array, y_array, z_array = calibration.get_mm(DETX0, params_dict_list, image_array)
data_plotting.plot_3d_mm(x_array, y_array, z_array, image_array_abs, index_list=INDEX_LIST)
