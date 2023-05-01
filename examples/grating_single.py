# examples/grating_single.py

import os
import numpy as np
from xray_scatter_py import data_plotting, utils, calibration, gratings


# 78465 - 78665 PHI = -5.5
# 78670 - 78870 PHI = 4.8
# 78875 - 79075 PHI = -0.04
# 79080 - 79280 PHI = 90

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_grating')

DETX0 = 100.4
INDEX_LIST = [0]

START_INDEX = 79035
PHI = -0.04

END_INDEX = START_INDEX


params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, START_INDEX, END_INDEX)
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.get_rel_intensity(
    params_dict_list, image_array, sr_array)

data_plotting.plot_2d(
    qy_array,
    qz_array,
    image_array_rel,
    index_list=INDEX_LIST)
qx, qy, qz, qx_0, qy_0, qz_0 = gratings.calculate_q(
    params_dict_list, image_array, phi=PHI, b=1390)
data_plotting.plot_2d_withmarkers(
    qy_array, qz_array, image_array, qy, qz, index_list=INDEX_LIST)


xmin, xmax = np.min(qy_array), np.max(qy_array)
ymin, ymax = np.min(qz_array), np.max(qz_array)

current_index = 0
for PHI in np.linspace(-10, 10, 5):
    qx, qy, qz, qx_0, qy_0, qz_0 = gratings.calculate_q(params_dict_list, image_array, phi=PHI, b=1390)
    data_plotting.plot_2d_onlymarkers(qy, qz, alpha=0.2, phi=PHI,
                                               xmin=np.min(qy_array), xmax=np.max(qy_array),
                                               ymin=np.min(qz_array), ymax=np.max(qz_array),
                                               write=True, current_index=current_index)
    current_index += 1
