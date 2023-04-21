# xray_scatter_py/data_processing.py

import numpy as np


def calculate_1d(q_min, q_max, q_num, q_array, image_array, omega, index_list=None):
    if index_list is None:
        index_list = [0]
    q_1d = np.linspace(q_min, q_max, q_num)
    q_fwhm = (q_max - q_min) / (q_num-1) / 2
    i_1d = np.empty((image_array.shape[0], q_num))
    for i in index_list:
        q_bool = np.abs(q_array[i][:, :, np.newaxis] - q_1d[np.newaxis, np.newaxis, :]) <= q_fwhm
        image_bool = image_array[i]!=-1
        sum_intensity = np.sum(image_array[i][:, :, np.newaxis] * omega[i][:, :, np.newaxis] * q_bool * image_bool[:, :, np.newaxis], axis=(0, 1))
        i_1d[i] = sum_intensity / np.sum(omega[i][:, :, np.newaxis] * q_bool * image_bool[:, :, np.newaxis], axis=(0, 1))
    return i_1d

def calculate_1d_cheap(q_min, q_max, q_num, q_array, image_array, omega, index_list=None):
    if index_list is None:
        index_list = [0]
    q_1d = np.linspace(q_min, q_max, q_num)
    q_fwhm = (q_max - q_min) / (q_num-1) / 2
    i_1d = np.empty((image_array.shape[0], q_num))
    for i in index_list:
        image_bool = image_array[i]!=-1
        for j in range(0, len(q_1d)):
            q_bool = np.abs(q_array[i] - q_1d[j]) <= q_fwhm
            sum_intensity = np.sum(image_array[i] * omega[i] * q_bool * image_bool, axis=(0, 1))
            i_1d[i][j] = (sum_intensity / np.sum(omega[i] * q_bool * image_bool, axis=(0, 1)))
    return i_1d

def calculate_1d_oop(qy_fwhm, qz_min, qz_max, qz_num, qy_array, qz_array, params_dict_list, image_array, omega, index_list=None):
    if index_list is None:
        index_list = [0]
    qz_1d = np.linspace(qz_min, qz_max, qz_num)
    qz_fwhm = (qz_max - qz_min) / (qz_num-1) / 2
    i_1d = np.empty((image_array.shape[0], qz_num))
    for i in index_list:
        qz_bool = np.abs(qz_array[i][:, :, np.newaxis] - qz_1d[np.newaxis, np.newaxis, :]) <= qz_fwhm
        image_bool = image_array[i]!=-1
        qy_bool = np.abs(qy_array[i]) <= qy_fwhm
        sum_intensity = np.sum(image_array[i][:, :, np.newaxis] * omega[i][:, :, np.newaxis] * qz_bool * image_bool[:, :, np.newaxis] * qy_bool[:, :, np.newaxis], axis=(0, 1))
        i_1d[i] = sum_intensity / np.sum(omega[i][:, :, np.newaxis] * qz_bool * image_bool[:, :, np.newaxis] * qy_bool[:, :, np.newaxis], axis=(0, 1))
    return i_1d