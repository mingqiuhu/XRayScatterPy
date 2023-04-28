# examples/sans.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sans')
FILE_NAME = 'EQSANS_139426_Iqxqy (1).dat'

data_array = utils.read_sans(DATA_PATH, FILE_NAME, header=3)
color_min = np.log10(np.min(data_array[:, 2][data_array[:, 2] > 0]))
color_max = np.log10(np.max(data_array[:, 2][data_array[:, 2] > 0]))

data_plotting.plot_2d_columns(data_array[:, 0], data_array[:, 1], data_array[:, 2])


START_INDEX = 51163
END_INDEX = START_INDEX
DETX0 = 100.4
INDEX_LIST = [0]

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.get_rel_intensity(
    params_dict_list, image_array, sr_array)

data_plotting.plot_2d(qy_array,
                                 qz_array,
                                 image_array_rel,
                                 index_list=INDEX_LIST,
                                 xticks=[-0.1,
                                         0,
                                         0.1,
                                         0.2],
                                 yticks=[0,
                                         0.05,
                                         0.1,
                                         0.15,
                                         0.2,
                                         0.25])
data_plotting.plot_2d_paralell(
    qx_array,
    qy_array,
    qz_array,
    image_array_rel,
    index_list=INDEX_LIST)
