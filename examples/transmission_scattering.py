# tests/test_data_plotting.py

import os
from xray_scatter_py import data_plotting, utils, calibration


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiff_files_calibration_q')
START_INDEX = 77023
END_INDEX = 77026
DETX0 = 100.4
INDEX_LIST = [0,1,2,3]

params_dict_list, image_array = utils.read_multiimage(DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.calculate_angle(DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.calculate_q(DETX0, params_dict_list, image_array)
omega = calibration.calculate_omega(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.calibrate_rel_intensity(params_dict_list, image_array, omega)
image_array_abs = calibration.calibrate_abs_intensity(params_dict_list, image_array_rel)
data_plotting.plot_2d_scattering(qy_array, qz_array, image_array_abs, index_list=INDEX_LIST, video=False)
data_plotting.plot_2d_polar(azimuth_array, qx_array, qy_array, qz_array, image_array_abs, index_list=INDEX_LIST)
data_plotting.plot_3d(qx_array, qy_array, qz_array, image_array_abs, index_list=INDEX_LIST)
# x_array, y_array, z_array = calibration.calculate_mm(DETX0, params_dict_list, image_array)
# data_plotting.plot_3d_mm(x_array, y_array, z_array, image_array_abs, index_list=INDEX_LIST)