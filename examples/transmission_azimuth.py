# examples/transmission_azimuth.py

import os
import numpy as np
from xray_scatter_py import data_processing, data_plotting, utils, calibration

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_calibration_q')
DETX0 = 100.4
INDEX = 81042
Q_TARGET = 1.5
Q_TOL = 0.1

params_dict_list, image_array = utils.read_multiimage(
    DATA_PATH, INDEX, INDEX)
theta_array, azimuth_array = calibration.get_angle(
    DETX0, params_dict_list, image_array)
qx_array, qy_array, qz_array = calibration.get_q(
    DETX0, params_dict_list, image_array)
q_array = np.sqrt(qx_array**2 + qy_array**2 + qz_array**2)
sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
image_array_rel = calibration.get_rel_intensity(
    params_dict_list, image_array, sr_array)

data_plotting.plot_2d(
    qy_array,
    qz_array,
    image_array_rel)
data_plotting.plot_2d_polar(
    azimuth_array,
    qx_array,
    qy_array,
    qz_array,
    image_array_rel,
    params_dict_list)
azmimuth_1d, i_1d = data_processing.calculate_1d_azimuth(
    q_array,
    azimuth_array,
    sr_array,
    image_array_rel,
    q_target=Q_TARGET,
    q_tol=Q_TOL)
data_plotting.plot_1d(azmimuth_1d, i_1d[0])

def calculate_order_param(azimuth: np.ndarray, intensity: np.ndarray) -> float:
    azimuth_rad = np.radians(azimuth)
    intensity = np.nan_to_num(intensity, nan=0.0)
    iop = np.sum(intensity * np.cos(azimuth_rad)**2) / np.sum(intensity)
    s = 0.5 * (3 * iop - 1)
    print("Order parameter (S) =", s)
    return s
calculate_order_param(azmimuth_1d, i_1d[0])