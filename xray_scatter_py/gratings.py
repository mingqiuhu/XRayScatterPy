# -*- coding: utf-8 -*-
# xray_scatter_py/gratings.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""This module provides functions to calculate the reciprocal space of a
grating in GISAXS experiments.

The main functions in this module are used to:

rotate_vector: A utility function to rotate a vector along an given axis by a
    given angle.
calculate_q: Calculate the reciprocal space lattice of a grating.
"""


import numpy as np

from xray_scatter_py import utils


def rotate_vector(
        vec: np.ndarray,
        axis: np.ndarray,
        theta_deg: float) -> np.ndarray:
    """Rotate a vector about an axis by a given angle.

    Args:
        - vec (np.ndarray): A 3-element array, the vector to rotate.
        - axis (np.ndarray): A 3-element array, the axis of rotation.
        - theta_deg (float): The angle of rotation in degrees.

    Returns:
        - np.ndarray: A 3-element array representing the rotated vector.
    """

    utils.validate_array_dimension(vec, 1)
    utils.validate_array_shape(vec, axis)

    theta_rad = np.radians(theta_deg)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cross_product_matrix = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
    rotation_matrix = (np.identity(3)*cos_theta +
                       (1 - cos_theta)*np.outer(axis, axis) +
                       sin_theta*cross_product_matrix)
    return np.dot(rotation_matrix, vec)


def calculate_q(params: list[dict],
                images: np.ndarray,
                **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the reciprocal space lattice of a grating.

    Args:
        - params (list[dict]): Each dict contains parameters of a measurement.
            Each dictionary must contain the following keys with string values:
            - 'wavelength': The wavelength of the x-ray beam.
            - 'sample_angle1': The incidence angle of x-ray beam on grating.
                The unit is degree.
        - images (np.ndarray): A 3D array of the original detector images.
            The first index is the serial number of measurement.
        - kwargs:
            - b (float): Full pitch of the grating, the unit is Å.
            - phi (float): The inplane rotation angle of the grating in degree.

    Returns: A tuple of six 2D arrays representing the reciprocal lattice. qx,
        qy, qz are the end points of the reciprocal lattice vectors, on the
        surface of the Ewald sphere. qx_0, qy_0, qz_0 are the beginning points
        of the reciprocal lattice vectors, on the realspace sample surface.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])
    utils.validate_kwargs({'b', 'phi'}, kwargs)

    b = kwargs['b']
    phi = np.radians(kwargs['phi'])

    # A list of reflection order of the reciprocal lattice.
    p = np.arange(-200, 200)

    # Initiallize the reciprocal lattice vectors. The vector points from
    # (qx_0, qy_0, qz_0) to (qx, qy, qz). Each of those six variables is a 2D
    # array with the first index being the serial number of measurement and
    # the second index being the reflection order.
    qz = np.empty((images.shape[0], p.shape[0]))
    qy = np.empty((images.shape[0], p.shape[0]))
    qx = np.empty((images.shape[0], p.shape[0]))
    qz_0 = np.empty((images.shape[0], p.shape[0]))
    qy_0 = np.empty((images.shape[0], p.shape[0]))
    qx_0 = np.empty((images.shape[0], p.shape[0]))

    # loop through each measurement
    for i in range(images.shape[0]):
        wavelength = float(params[i]['wavelength'])
        alpha = np.radians(float(params[i]['sample_angle1']))

        # calculate the end point of the reciprocal lattice vectors
        qz[i] = (2 * np.pi * np.sin(alpha) * np.cos(alpha) * np.cos(phi) /
                 wavelength * (1 + (1 - wavelength**2 * p**2 / b**2 /
                                    np.sin(alpha)**2 / np.cos(phi)**2
                                    * (1 - 2 * b * np.sin(phi) / p /
                                       wavelength))**0.5))
        qy[i] = (2 * np.pi * p * np.cos(phi) / b - 2 * np.pi *
                 np.sin(alpha)**2 * np.cos(phi) * np.sin(phi) / wavelength *
                 (1 + (1 - wavelength**2 * p**2 / b**2 / np.sin(alpha)**2 /
                       np.cos(phi)**2 * (1 - 2 * b * np.sin(phi) / p /
                                         wavelength))**0.5))
        qx[i] = -np.sqrt((2 * np.pi / wavelength)**2 - qy[i] **
                         2 - qz[i]**2) + 2 * np.pi / wavelength

        # calculate the beginning point of the reciprocal lattice vectors at 0
        # inplane rotation
        qx_0[i] = 0
        qy_0[i] = 2 * np.pi * p / b
        qz_0[i] = 0
        # calculate the beginning point of the reciprocal lattice vectors at
        # phi inplane rotation
        for j in range(p.shape[0]):
            vec = np.array([qx_0[i][j], qy_0[i][j], qz_0[i][j]])
            axis = np.array([np.sin(alpha), 0, np.cos(alpha)])
            theta_deg = phi
            new_vec = rotate_vector(vec, axis, theta_deg)
            qx_0[i][j] = new_vec[0]
            qy_0[i][j] = new_vec[1]
            qz_0[i][j] = new_vec[2]

        # mask out the invalid points with the limit of the available
        # diffraction orders
        p_upper_limit = b * np.sin(phi) / wavelength + b / wavelength * \
            (np.sin(phi)**2 + np.sin(alpha)**2 * np.cos(phi)**2)**0.5
        p_lower_limit = b * np.sin(phi) / wavelength - b / wavelength * \
            (np.sin(phi)**2 + np.sin(alpha)**2 * np.cos(phi)**2)**0.5
        mask = np.logical_and(p <= p_upper_limit, p >= p_lower_limit)
        qx[i, ~mask] = np.nan
        qy[i, ~mask] = np.nan
        qz[i, ~mask] = np.nan
        qx_0[i, ~mask] = np.nan
        qy_0[i, ~mask] = np.nan
        qz_0[i, ~mask] = np.nan

    return qx, qy, qz, qx_0, qy_0, qz_0
