# -*- coding: utf-8 -*-
# xray_scatter_py/penetration.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""Calculate the penetration of the x-ray or neutron beam in the material.

The main functions in this module are:

calc_alpha_c: Calculate critical angle.
calc_depth: Calculate the scattering depth.
calc_lilf: Calculate the penetration of the incident or exit beam.
calc_alpha_array: Build an array of input parameters.
calc_sym_depth: Calculate the scattering depth for symmetric geometry.
calc_asym_depth: Calculate the scattering depth for asymmetric geometry?
analyze_symmetric_penetration: Analyze the penetration for symmetric geometry?
"""
import numpy as np
import matplotlib.pyplot as plt

from xray_scatter_py import data_plotting, utils


def calc_alpha_c(wavelength: float, sld: float) -> float:
    """Calculate critical angle.

    Args:
        - wavelength (float): wavelength of x-ray or neutron.
        - sld (float): real scattering length density.

    Returns:
        - float: critical angle of the material in ratian.
    """
    return np.arcsin(wavelength * np.sqrt(sld / np.pi))


def calc_depth(wavelength: float, li: float, lf: float) -> float:
    """Calculate the scattering depth.

    Args:
        - wavelength (float): wavelength of x-ray or neutron.
        - li (float): Contribution of the incidence beam penetration.
        - lf (float): Contribution of the exit beam penetration.

    Returns:
        - float: the scattering depth.
    """
    return wavelength / np.sqrt(2) / np.pi / (li + lf)


def calc_lilf(alpha_c: float,
              alpha_if: float,
              wavelength: float,
              sldi: float) -> float:
    """Calculate the penetration of the incident or exit beam.

    Args:
        - alpha_c (float): critical angle of the material in radian.
        - alpha_if (float): incident or exit angle of the beam in radian.
        - wavelength (float): wavelength of x-ray or neutron.
        - sldi (float): imaginary or incoherant scattering length density.

    Returns:
        - float: the penetration of the incident or exit beam.
    """
    return np.sqrt(
        alpha_c**2 - alpha_if**2 + np.sqrt((alpha_c**2 - alpha_if**2)**2 +
                                           4*(wavelength**2/2/np.pi*sldi)**2))


def calc_sym_depth(wavelength: float,
                   sld: float,
                   sldi: float,
                   alpha_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the scattering depth for symmetric geometry.

    Args:
        - wavelength (float): wavelength of x-ray or neutron.
        - sld (float): real scattering length density.
        - sldi (float): imaginary or incoherant scattering length density.
        - alpha_array (np.ndarray): an array of incident or exit angle of the
            beam in radian.

    Returns:
        - tuple(np.ndarray, np.ndarray): the scattering depth for symmetric
            geometry. The first is 1d array, the second is 2d mesh.
    """
    alpha_c = calc_alpha_c(wavelength, sld)
    l_list = calc_lilf(alpha_c, alpha_array, wavelength, sldi)
    depth_1d = calc_depth(wavelength, l_list, l_list)
    depth_mesh = calc_depth(
        wavelength, l_list.reshape((-1, 1)), l_list.reshape((1, -1)))
    return depth_1d, depth_mesh


def calc_asym_depth(wavelength: float,
                    sld: float,
                    sldi: float,
                    alpha_f_array: np.ndarray,
                    alpha_i: float) -> np.ndarray:
    """Calculate the scattering depth for in a real grazing incident geometry.

    Args:
        - wavelength (float): wavelength of x-ray or neutron.
        - sld (float): real scattering length density.
        - sldi (float): imaginary or incoherant scattering length density.
        - alpha_f_array (np.ndarray): an array of exit angle of the beam in
            radian.
        - alpha_i (float): incident angle of the beam in radian.

    Returns:
        - np.ndarray: 1D array of the scattering depth for asymmetric geometry.
    """

    alpha_c = calc_alpha_c(wavelength, sld)
    li = calc_lilf(alpha_c, alpha_i, wavelength, sldi)
    lf_list = calc_lilf(alpha_c, alpha_f_array, wavelength, sldi)
    depth_1d = calc_depth(wavelength, li, lf_list)
    return depth_1d


def analyze_symmetric_penetration(**kwargs) -> None:
    """Analyze the scattering depth of a real system in symmetric geometry.

    Args:
        - kwargs: a dictionary of input parameters.
            - num (int): number of points in the alpha_array.
            - wavelength (float): wavelength of x-ray or neutron.
            - sld (float): real scattering length density.
            - sldi (float): imaginary or incoherant scattering length density.
            - legend (list[str]): legend of the plot.
            - x_max_1d (float): maximum value of the x-axis in 1D plot.
            - y_min_1d (float): minimum value of the y-axis in 1D plot.
            - y_max_1d (float): maximum value of the y-axis in 1D plot.
            - x_max_2d (float): maximum value of the x-axis in 2D plot.
            - y_max_2d (float): maximum value of the y-axis in 2D plot.

    Returns:
        - None
    """

    utils.validate_kwargs({'num', 'wavelength', 'sld', 'sldi', 'legend',
                           'x_max_1d', 'y_min_1d', 'y_max_1d', 'x_max_2d',
                           'y_max_2d'}, kwargs)
    num = kwargs['num']
    wavelength = kwargs['wavelength']
    sld = kwargs['sld']
    sldi = kwargs['sldi']
    legend = kwargs['legend']
    x_max_1d = kwargs['x_max_1d']
    y_min_1d = kwargs['y_min_1d']
    y_max_1d = kwargs['y_max_1d']
    x_max_2d = kwargs['x_max_2d']
    y_max_2d = kwargs['y_max_2d']

    radius_max = np.arcsin(
        2 * max(x_max_1d, x_max_2d, y_max_2d) * wavelength / 4 / np.pi)

    alpha_array = np.linspace(0, radius_max, num)
    depth_1d, depth_mesh = calc_sym_depth(
        wavelength, sld, sldi, alpha_array)
    qz_1d = 4 * np.pi * np.sin(alpha_array) / wavelength
    data_plotting.plot_1d_sdepth(
        qz_1d,
        depth_1d,
        x_max=x_max_1d,
        y_min=y_min_1d,
        y_max=y_max_1d,
        legend=legend)
    alpha_i_mesh, alpha_f_mesh = np.meshgrid(alpha_array, alpha_array)
    ki_mesh = 2 * np.pi * np.sin(alpha_i_mesh) / wavelength
    kf_mesh = 2 * np.pi * np.sin(alpha_f_mesh) / wavelength
    data_plotting.plot_2d_sdepth(
        ki_mesh, kf_mesh, depth_mesh, x_max=x_max_2d, y_max=y_max_2d)


def analyze_assymetric_penetration(**kwargs) -> None:
    """Analyze the scattering depth of a real system in grazing incidence.

    Args:
        - kwargs: a dictionary of input parameters.
            - num (int): number of points in the alphaf_array.
            - wavelength (float): wavelength of x-ray or neutron.
            - sld (float): real scattering length density.
            - sldi (float): imaginary or incoherant scattering length density.
            - alpha_i_array (np.ndarray): array of the incident angle in degree
            - legend (list[str]): legend of the plot.
            - x_max_1d (float): maximum value of the x-axis in 1D plot.
            - y_min_1d (float): minimum value of the y-axis in 1D plot.
            - y_max_1d (float): maximum value of the y-axis in 1D plot.

    Returns:
        - None
    """
    num = kwargs['num']
    wavelength = kwargs['wavelength']
    sld = kwargs['sld']
    sldi = kwargs['sldi']
    alpha_i_array = kwargs['alpha_i_array']
    legend = kwargs['legend']
    x_max_1d = kwargs['x_max_1d']
    y_min_1d = kwargs['y_min_1d']
    y_max_1d = kwargs['y_max_1d']

    alpha_i_radius_array = np.radians(alpha_i_array)
    radius_max = np.arcsin(2 * x_max_1d * wavelength / 4 / np.pi) * \
        2 - np.min(alpha_i_radius_array)
    alpha_f_array = np.linspace(0, radius_max, num)
    data_plotting.plot_set()
    for i in range(alpha_i_array.shape[0]):
        depth_1d = calc_asym_depth(
            wavelength, sld, sldi, alpha_f_array, alpha_i_radius_array[i])
        qz_1d = 4 * np.pi * \
            np.sin((alpha_f_array + alpha_i_radius_array[i]) / 2) / wavelength
        plt.plot(qz_1d, depth_1d)
    plt.yscale('log')
    plt.xlim(0, x_max_1d)
    plt.ylim(y_min_1d, y_max_1d)
    plt.ylabel(r'$z_\mathrm{1/e}\ \mathrm{(Å)}$')
    plt.xlabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$')
    plt.legend(legend, fontsize=19)
    plt.show()
