# xray_scatter_py/gratings.py

import numpy as np


def rotate_vector(vec, axis, theta_deg):
    theta_rad = np.radians(theta_deg)
    axis = axis / np.linalg.norm(axis)  # normalize the rotation axis
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cross_product_matrix = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
    rotation_matrix = (np.identity(3) * cos_theta +
                       (1 - cos_theta) * np.outer(axis, axis) +
                       sin_theta * cross_product_matrix)
    return np.dot(rotation_matrix, vec)


def calculate_q(params_dict_list, image_array, b=1390, phi=None):
    phi = np.radians(phi)
    p = np.arange(-60, 61)
    qz = np.empty((image_array.shape[0], p.shape[0]))
    qy = np.empty((image_array.shape[0], p.shape[0]))
    qx = np.empty((image_array.shape[0], p.shape[0]))
    qz_0 = np.empty((image_array.shape[0], p.shape[0]))
    qy_0 = np.empty((image_array.shape[0], p.shape[0]))
    qx_0 = np.empty((image_array.shape[0], p.shape[0]))
    for i in range(image_array.shape[0]):
        wavelength = float(params_dict_list[i]['wavelength'])
        alpha = np.radians(float(params_dict_list[i]['sample_angle1']))
        qz[i] = 2 * np.pi * np.sin(alpha) * np.cos(alpha) * np.cos(phi) / wavelength \
        * (1 + (1 - wavelength**2 * p**2 / b**2 / np.sin(alpha)**2 / np.cos(phi)**2
                * (1 - 2 * b * np.sin(phi) / p / wavelength))**0.5)
        qy[i] = 2 * np.pi * p * np.cos(phi) / b - 2 * np.pi * np.sin(alpha)**2 * np.cos(phi) * np.sin(phi) / wavelength * \
        (1 + (1 - wavelength**2 * p**2 / b**2 / np.sin(alpha)**2 / np.cos(phi)**2 * (1 - 2 * b * np.sin(phi) / p / wavelength))**0.5)
        qx[i] = -np.sqrt((2*np.pi/wavelength)**2 - qy[i]**2 - qz[i]**2) + 2*np.pi/wavelength
        
        qx_0[i] = 0
        qy_0[i] = 2 * np.pi * p / b
        qz_0[i] = 0
        
        for j in range(p.shape[0]):
            vec = np.array([qx_0[i][j], qy_0[i][j], qz_0[i][j]])
            axis = np.array([np.sin(alpha), 0, np.cos(alpha)])
            theta_deg = phi
            new_vec = rotate_vector(vec, axis, theta_deg)
            qx_0[i][j], qy_0[i][j], qz_0[i][j] = new_vec[0], new_vec[1], new_vec[2]
        
    
        p_upper_limit = b * np.sin(phi) / wavelength + b / wavelength * (np.sin(phi)**2 + np.sin(alpha)**2 * np.cos(phi)**2)**0.5
        p_lower_limit = b * np.sin(phi) / wavelength - b / wavelength * (np.sin(phi)**2 + np.sin(alpha)**2 * np.cos(phi)**2)**0.5
        mask = np.logical_and(p <= p_upper_limit, p >= p_lower_limit)
        qx[i, ~mask] = np.nan
        qy[i, ~mask] = np.nan
        qz[i, ~mask] = np.nan
        qx_0[i, ~mask] = np.nan
        qy_0[i, ~mask] = np.nan
        qz_0[i, ~mask] = np.nan
    return qx, qy, qz, qx_0, qy_0, qz_0