import numpy as np
from xray_scatter_py import data_plotting
import matplotlib.pyplot as plt


def calc_alpha_c(wavelength, sld):
    return np.arcsin(wavelength * np.sqrt(sld / np.pi))


def calc_depth(wavelength, li, lf):
    return wavelength / np.sqrt(2) / np.pi / (li + lf)


def calc_lilf(alpha_c, alpha_if, wavelength, sldi):
    return np.sqrt(alpha_c**2 - alpha_if**2 + np.sqrt((alpha_c**2 -
                   alpha_if**2)**2 + 4 * (wavelength**2 / 2 / np.pi * sldi)**2))


def calc_alpha_array(radius_max, num):
    return np.linspace(0, radius_max, num)


def calc_sym_depth(wavelength, sld, sldi, alpha_array):
    alpha_c = calc_alpha_c(wavelength, sld)
    l_list = calc_lilf(alpha_c, alpha_array, wavelength, sldi)
    depth_1d = calc_depth(wavelength, l_list, l_list)
    depth_mesh = calc_depth(wavelength, l_list.reshape(
        (-1, 1)), l_list.reshape((1, -1)))
    return depth_1d, depth_mesh


def calc_asym_depth(wavelength, sld, sldi, alpha_f_array, alpha_i):
    alpha_c = calc_alpha_c(wavelength, sld)
    li = calc_lilf(alpha_c, alpha_i, wavelength, sldi)
    lf_list = calc_lilf(alpha_c, alpha_f_array, wavelength, sldi)
    depth_1d = calc_depth(wavelength, li, lf_list)
    return depth_1d


def analyze_symmetric_penetration(
        num,
        wavelength,
        sld,
        sldi,
        legend,
        x_max_1d,
        y_min_1d,
        y_max_1d,
        x_max_2d,
        y_max_2d):
    radius_max = np.arcsin(
        2 * max(x_max_1d, x_max_2d, y_max_2d) * wavelength / 4 / np.pi)

    alpha_array = calc_alpha_array(radius_max, num)
    depth_1d, depth_mesh = calc_sym_depth(
        wavelength, sld, sldi, alpha_array)
    qz_1d = 4 * np.pi * np.sin(alpha_array) / wavelength
    data_plotting.plot_penetration(
        qz_1d,
        depth_1d,
        legend,
        x_max=x_max_1d,
        y_min=y_min_1d,
        y_max=y_max_1d)
    alpha_i_mesh, alpha_f_mesh = np.meshgrid(alpha_array, alpha_array)
    ki_mesh = 2 * np.pi * np.sin(alpha_i_mesh) / wavelength
    kf_mesh = 2 * np.pi * np.sin(alpha_f_mesh) / wavelength
    data_plotting.plot_penetration_2d(
        ki_mesh, kf_mesh, depth_mesh, x_max=x_max_2d, y_max=y_max_2d)


def analyze_assymetric_penetration(
        num,
        wavelength,
        sld,
        sldi,
        alpha_i_array,
        legend,
        x_max_1d,
        y_min_1d,
        y_max_1d):
    alpha_i_radius_array = np.radians(alpha_i_array)
    radius_max = np.arcsin(2 * x_max_1d * wavelength / 4 / np.pi) * \
        2 - np.min(alpha_i_radius_array)
    alpha_f_array = calc_alpha_array(radius_max, num)
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
