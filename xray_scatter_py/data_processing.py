# -*- coding: utf-8 -*-
# xray_scatter_py/data_processing.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""This module provides functions to process the 2D detector images with
calibration to calculate 1D intensity as a funciton of scatterin vector q.

The main functions in this module are used to:

1. calculate_1d: Calculates the 1D intensity as a function of q (Å^-1) from a
    2D image array. The intensity is averaged over all the azimuthal angles at
    each q value.
2. calculate_1d_lowmemo: Similar to calculate_1d but designed for use with
    limited memory.
3. calculate_1d_oop: Calculates the 1D scattering intensity in the out-of-
    plane direction at qy=0, for grazing indicence x-ray scattering, with I as
    a function of qz (Å*^-1).

Each function takes the following inputs:
q_array: a 3D array of q values corresponding to every pixel on the detector
images: a 3D array of detector images, either relative or absolute intensity
sr_array: a 3D array of solid angles corresponding to every pixel on detector.

These functions return 1D scattering intensity profile in numpy array.
"""


import numpy as np

from xray_scatter_py import utils


def calculate_1d(
        q_array: np.ndarray,
        images: np.ndarray,
        sr_array: np.ndarray,
        **kwargs) -> np.ndarray:
    """Calculate the 1D scattering intensity from a 2D image array.

    Args:
        - q_array (np.ndarray): 3D array of q of every detector pixel.
            The first index is the serial number of measurement. The second and
            third indices are the y and z indices of the detector image.
        - images (np.ndarray): A 3D array of the original detector images.
            The first index is the serial number of measurement.
        - sr_array (np.ndarray): A 3D array of the solid angles for each pixel.
        - kwargs:
            - q_min (float, optional): minimum q value in the 1D plots in Å^-1.
                If not provided, default to 0.
            - q_max (float, optional): maximum q value in the 1D plots,in Å^-1.
                If not provided, default to 3.
            - q_num (int, optional): number of q values in the 1D plots.
                If not provided, default to 400.
            - index_list (list[int], optional): list of indexes to process.
                If not provided, defaults to [0].

    Returns:
        - np.ndarray: 2D array of averaged scattering intensity. The first
            index is the serial number of measurement. The second index is the
            intensity at each q.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_array_shape(q_array, images, sr_array)
    utils.validate_kwargs({'q_min', 'q_max', 'q_num', 'index_list'}, kwargs)

    q_min = kwargs.get('q_min', 0)
    q_max = kwargs.get('q_max', 3)
    q_num = kwargs.get('q_num', 400)
    index_list = kwargs.get('index_list', [0])

    # calculate the q values to be used in the 1D intensity profile
    q_1d = np.linspace(q_min, q_max, q_num)
    # for each q value, the range used to calculate the intensity is q-q_fwhm
    # to q+q_fwhm
    q_fwhm = (q_max - q_min) / (q_num - 1) / 2

    # initialize the 1D intensity array
    # the first index is the serial number of measurement, the second index
    # relates to the q value
    i_1d = np.empty((images.shape[0], q_num))

    # loop over all the serial numbers of measurements to be processed
    for i in index_list:
        # broadcast the 2D q array to 3D and generate a boolean mask for 1D
        # averaging. The third dimension is the same as the 1D q array for 1D
        # intensity profile
        q_bool = np.abs(q_array[i][:, :, np.newaxis] -
                        q_1d[np.newaxis, np.newaxis, :]) <= q_fwhm
        # generate a boolean mask for the image array excluding the pixels with
        # -1 values
        image_bool = images[i] != -1
        # Since relative or absolute intensity is normalized by solid angle,
        # the sum of the intensity image needs to be weighted by solid angle.
        # after sum, the output is a 1D array with the same length as the 1D q
        # array
        sum_intensity = np.sum(images[i][:, :, np.newaxis] *
                               sr_array[i][:, :, np.newaxis] *
                               q_bool *
                               image_bool[:, :, np.newaxis],
                               axis=(0, 1))
        # assign the 1D intensity profile to the 1D intensity array after
        # normalization with the total solid angle of the pixels used in sum
        i_1d[i] = sum_intensity / np.sum(sr_array[i][:, :, np.newaxis] *
                                         q_bool * image_bool[:, :, np.newaxis],
                                         axis=(0, 1))
    return i_1d


def calculate_1d_lowmemo(
        q_array: np.ndarray,
        images: np.ndarray,
        sr_array: np.ndarray,
        **kwargs) -> np.ndarray:
    """Calculate the 1D scattering intensity from a 2D image array.
    This function is designed to be used when memory is limited.

    Args:
        - q_array (np.ndarray): 3D array of q of every detector pixel.
            The first index is the serial number of measurement. The second and
            third indices are the y and z indices of the detector image.
        - images (np.ndarray): A 3D array of the original detector images.
            The first index is the serial number of measurement.
        - sr_array (np.ndarray): A 3D array of the solid angles for each pixel.
        - kwargs:
            - q_min (float, optional): minimum q value in the 1D plots in Å^-1.
                If not provided, default to 0.
            - q_max (float, optional): maximum q value in the 1D plots,in Å^-1.
                If not provided, default to 3.
            - q_num (int, optional): number of q values in the 1D plots.
                If not provided, default to 400.
            - index_list (list[int], optional): list of indexes to process.
                If not provided, defaults to [0].

    Returns:
        - np.ndarray: 2D array of averaged scattering intensity. The first
            index is the serial number of measurement. The second index is the
            intensity at each q.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_array_shape(q_array, images, sr_array)
    utils.validate_kwargs({'q_min', 'q_max', 'q_num', 'index_list'}, kwargs)

    q_min = kwargs.get('q_min', 0)
    q_max = kwargs.get('q_max', 3)
    q_num = kwargs.get('q_num', 400)
    index_list = kwargs.get('index_list', [0])

    # calculate the q values to be used in the 1D intensity profile
    q_1d = np.linspace(q_min, q_max, q_num)
    # for each q value, the range used to calculate the intensity is q-q_fwhm
    # to q+q_fwhm
    q_fwhm = (q_max - q_min) / (q_num - 1) / 2

    # initialize the 1D intensity array
    # the first index is the serial number of measurement, the second index
    # relates to the q value
    i_1d = np.empty((images.shape[0], q_num))

    # loop over all the serial numbers of measurements to be processed
    for i in index_list:
        # generate a boolean mask for the image array excluding the pixels with
        # -1 values
        image_bool = images[i] != -1
        # loop through all the q values in the 1D q array for calculating the
        # 1D intensity array
        for i_q_curr, q_curr in enumerate(q_1d):
            # generate a boolean mask for 1D intensity averaging with same
            # shape of the 2D q_array
            q_bool = np.abs(q_array[i] - q_curr) <= q_fwhm
            # calculate integrated intensity at one q value for one measurement
            # the sum of the intensity image needs to be weighted by the solid
            # angle.
            sum_intensity = np.sum(images[i] *
                                   sr_array[i] *
                                   q_bool *
                                   image_bool,
                                   axis=(0, 1))
            # assign the 1D intensity to the 1D intensity array after
            # normalization with total solid angle used in sum
            i_1d[i][i_q_curr] = (sum_intensity / np.sum(sr_array[i] *
                                                        q_bool *
                                                        image_bool,
                                                        axis=(0, 1)))
    return i_1d


def calculate_1d_oop(
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        sr_array: np.ndarray,
        **kwargs) -> np.ndarray:
    """Calculate the 1D scattering intensity in out-of-plane direction at qy=0.

    Args:
        - qy_array (np.ndarray): 3D array of qy values for each detector pixel.
            The first index is the serial number of measurement.
        - qz_array (np.ndarray): 3D array of qz values for each detector pixel.
        - images (np.ndarray): A 3D array of the original detector images.
            The first index is the serial number of measurement.
        - sr_array (np.ndarray): A 3D array of the solid angles for each pixel.
        - kwargs:
            - qy_fwhm (float, optional): integral range is -qy_fwhm to qy_fwhm.
            - qz_min (float, optional): minimum qz in 1D profile, in Å^-1.
            - qz_max (float, optional): maximum qz in 1D profile, in Å^-1.
            - qz_num (int, optional): number of qz in 1D profile.
            - index_list (list[int], optional): list of indexes to process.
                If not provided, defaults to [0].
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_array_shape(qy_array, qz_array, images, sr_array)
    utils.validate_kwargs({'qy_fwhm', 'qz_min', 'qz_max',
                          'qz_num', 'index_list'}, kwargs)

    qy_fwhm = kwargs.get('qy_fwhm', 0.002)
    qz_min = kwargs.get('qz_min', 0)
    qz_max = kwargs.get('qz_max', 0.2)
    qz_num = kwargs.get('qz_num', 400)
    index_list = kwargs.get('index_list', [0])

    # calculate the qz values to be used in the 1D intensity profile
    qz_1d = np.linspace(qz_min, qz_max, qz_num)
    # for each qz value, the range used to calculate the intensity is
    # qz-q_fwhm to qz+q_fwhm
    qz_fwhm = (qz_max - qz_min) / (qz_num - 1) / 2
    # initialize the 1D intensity array
    # the first index is the serial number of measurement, the second index
    # relates to the qz value
    i_1d = np.empty((images.shape[0], qz_num))

    # loop over all the serial numbers of measurements to be processed
    for i in index_list:
        # generate a 3D boolean mask with the same dimension as the qy_array
        # representing the qy range used to calculate the 1D intensity profile
        qy_bool = np.abs(qy_array[i]) <= qy_fwhm
        # broadcast 2D q array to 3D and generate a boolean mask for averaging
        # the third dimension is the same as the 1D qz array for the 1D
        # intensity profile
        qz_bool = np.abs(qz_array[i][:, :, np.newaxis] -
                         qz_1d[np.newaxis, np.newaxis, :]) <= qz_fwhm
        # generate a boolean mask for the image array excluding the pixels with
        # -1 values
        image_bool = images[i] != -1
        # Since relative or absolute intensity is normalized by solid angle,
        # the sum of the intensity image needs to be weighted by solid angle.
        # after sum, the output is a 1D array with the same length as the 1D q
        # array
        sum_intensity = np.sum(images[i][:, :, np.newaxis] *
                               sr_array[i][:, :, np.newaxis] *
                               qz_bool * image_bool[:, :, np.newaxis] *
                               qy_bool[:, :, np.newaxis],
                               axis=(0, 1))
        # assign the 1D intensity profile to the 1D intensity array after
        # normalization with the total solid angle of the pixels used in sum
        i_1d[i] = sum_intensity / np.sum(sr_array[i][:, :, np.newaxis] *
                                         qz_bool *
                                         image_bool[:, :, np.newaxis] *
                                         qy_bool[:, :, np.newaxis],
                                         axis=(0, 1))
    return i_1d
