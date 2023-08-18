# -*- coding: utf-8 -*-
# xray_scatter_py/data_plotting.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""This module provides functions to plot the processed scattering data.

The main functions in this module are used to:

Set the global matplotlib settings (plot_set, plot_set_small).
Plot 2D scattering colormaps with various projections. (plot_2d,
    plot_2d_withmarkers, plot_2d_onlymarkers, plot_2d_polar).
Plot 2D grazing incidence scattering colormaps in q_parallel axis
    (plot_2d_gi).
Plot and compare 1D scattering and reflectivity data (plot_1d, plot_1d_compare)
plot scattering colormaps in 3d real space or on the an Ewald sphere
    (plot_3d_q, plot_3d_mm, plot_3d_grating)
Plot scattering data given in columns instead of meshes, for example,
    neutron scattering data from ORNL, into colormap (plot_2d_columns).
Plot 1D penentration depth in a grazing incidence experiment
    (plot_1d_sdepth, plot_1d_sdepth_compare).
Plot 2D penentration colormaps in a grazig incidence experiment
    (plot_2d_sdepth).

These functions take in various q arrays and intensity arrays calculated and
processed by other modules like calibrations.py and dataprocessing.py.
These functions don't have typical return values. They create plots.
It is highly recommended run those functions in an interactive window.
"""

import numpy as np
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

from xray_scatter_py import utils

XLABEL_DICT = {
    'q': r'$q\ \mathrm{(Å^{-1})}$',
    'qx': r'$q_\mathrm{x}\ \mathrm{(Å^{-1})}$',
    'qy': r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$',
    'qz': r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$',
    'qz0': r'$q_\mathrm{z,0}\ \mathrm{(Å^{-1})}$',
    'q_parallel': r'$q_\Vert\ \mathrm{(Å^{-1})}$',
    'theta sample': r'${\theta}_\mathrm{sample}\ (°)$',
    'kzi': r'$k_\mathrm{z,i}\ \mathrm{(Å^{-1})}$',
    'alpha i': r'$\alpha_\mathrm{i}\ (°)$',
}
YLABEL_DICT = {
    'qz': r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$',
    'azimuth': r'$\mathrm{azimuth}\ (°)$',
    'abs': r'$I\ \mathrm{(cm^{-1}sr^{-1})}$',
    'a.u.': r'$I\ \mathrm{(a.u.)}$',
    'reflectivity': r'$R$',
    'total': r'$I_\mathrm{total}\ \mathrm{(a.u.)}$',
    'spillover': r'$I_\mathrm{spill\ over}\ \mathrm{(a.u.)}$',
    'depth': r'$z_\mathrm{1/e}\ \mathrm{(Å)}$',
    'kzf': r'$k_\mathrm{z,f}\ \mathrm{(Å^{-1})}$',
    'q_vertical': r'$q_\mathrm{⊥}\ \mathrm{(Å^{-1})}$'
}


def plot_set() -> None:
    """Update the global matplotlib settings with following parameters for
    higher figure resolution, larger tick labels, and thicker lines.

    Args:
        - None

    Returns:
        - None
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['xtick.labelsize'] = 22
    matplotlib.rcParams['ytick.labelsize'] = 22
    matplotlib.rcParams['font.size'] = 22
    matplotlib.rcParams['legend.fontsize'] = 22
    matplotlib.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['xtick.major.width'] = 3
    matplotlib.rcParams['xtick.minor.width'] = 3
    matplotlib.rcParams['ytick.major.width'] = 3
    matplotlib.rcParams['ytick.minor.width'] = 3


def plot_set_small() -> None:
    """Update the global matplotlib settings with following parameters for
    higher figure resolution, medium tick labels, and thicker lines.

    Args:
        - None

    Returns:
        - None
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['figure.dpi'] = 600
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['legend.fontsize'] = 16
    matplotlib.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['xtick.major.width'] = 3
    matplotlib.rcParams['xtick.minor.width'] = 3
    matplotlib.rcParams['ytick.major.width'] = 3
    matplotlib.rcParams['ytick.minor.width'] = 3


def plot_2d(
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        **kwargs) -> None:
    """This function takes qy_array, qz_array, and images as inputs and
    creates a 2D colormap plot for each index in index_list using the
    provided arrays. qx is ignored when scatering data is plotted in this way.

    Args:
        - qy_array (np.ndarray): 3D array of qy values (in Å^-1 units).
        - qz_array (np.ndarray): 3D array of qz values (in Å^-1 units).
        - images (np.ndarray): 3D array of the original detector images.
            The first index is the serial number of measurement.
        - kwargs:
            - index_list (list[int], optional): list of indexes of the
                measurements to plot. If not provided, defaults to [0].
            - crop (bool, optional): decide whether the plot would be cropped
                after correcting the chi rotation in a grazing incidence
                measurement. If not providded, default to False.
            - xticks (list[float], optional): set the xtick locations.
                If not provided, default to None
            - yticks (list[float], optional): set the ytick locations.
                If not provided, default to None
            - video (bool, optional): decide if the images would be used to
                generate videos. If not provided, default to False.

    Returns:
        - None
    """
    utils.validate_array_dimension(images, 3)
    utils.validate_array_shape(images, qy_array, qz_array)
    utils.validate_kwargs(
        {'index_list', 'crop', 'xticks', 'yticks', 'video'}, kwargs)

    index_list = kwargs.get('index_list', [0])
    crop = kwargs.get('crop', False)
    xticks = kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    video = kwargs.get('video', False)

    plot_set()
    for i in index_list:
        plt.figure()
        zmax = np.max(images[i])
        zmin = np.min(images[i][images[i] > 0])
        # zmin = max(zmax*1e-6, np.min(images[i][images[i] > 0]))
        norm = matplotlib.colors.LogNorm(zmin, zmax)
        plt.pcolormesh(
            qy_array[i],
            qz_array[i],
            images[i],
            cmap='jet',
            linewidths=3,
            norm=norm,
            shading='nearest')
        plt.xlabel(XLABEL_DICT['qy'])
        plt.ylabel(YLABEL_DICT['qz'])
        if not video:
            plt.colorbar(label='I (a.u.)')
        plt.gca().set_aspect('equal', adjustable='box')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        if crop:
            plt.xlim(
                qy_array[i][np.where(-qz_array[i] == np.max(-qz_array[i]))],
                qy_array[i][np.where(-qz_array[i] == np.min(-qz_array[i]))])
            plt.ylim(
                qz_array[i][np.where(-qy_array[i] == np.min(-qy_array[i]))],
                qz_array[i][np.where(-qy_array[i] == np.max(-qy_array[i]))])
        if video:
            plt.xlim(np.min(qy_array), np.max(qy_array))
            plt.ylim(np.min(qz_array), np.max(qz_array))
#        plt.savefig('2d.png', dpi=400, bbox_inches='tight')
        plt.show()

def plot_2d_withmarkers(
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        qy_markers_array: np.ndarray,
        qz_markers_array: np.ndarray,
        **kwargs) -> None:
    """This function takes qy_array, qz_array, and images as inputs and
    creates a 2D colormap plot with markers, for each index in index_list using
    provided qy_markers_array and qz_markers_array.

    Args:
        - qy_array (np.ndarray): 3D array of qy values (in Å^-1 units).
        - qz_array (np.ndarray): 3D array of qz values (in Å^-1 units).
        - images (np.ndarray): 3D array of scattering intensities.
            The first index is the serial number of measurement.
        - qy_markers_array (np.ndarray): 2D array of qy of the markers.
        - qz_markers_array (np.ndarray): 2D array of qz of the markers.
            The first index is the serial number of measurement.
        - kwargs:
            - index_list (list[int], optional): list of indexes to plot.
                If not provided, defaults to [0].
            - crop (bool, optional): decide whether the plot would be cropped
                after correcting the chi rotation in a grazing incidence
                measurement. If not providded, default to False.

    Returns:
        - None
    """
    utils.validate_array_dimension(images, 3)
    utils.validate_array_dimension(qy_markers_array, 2)
    utils.validate_array_shape(images, qy_array, qz_array)
    utils.validate_array_shape(qy_markers_array, qz_markers_array)
    utils.validate_kwargs({'index_list', 'crop'}, kwargs)

    index_list = kwargs.get('index_list', [0])
    crop = kwargs.get('crop', False)

    plot_set()
    for i in index_list:
        plt.figure()
        plt.pcolormesh(
            qy_array[i],
            qz_array[i],
            images[i],
            cmap='jet',
            linewidths=3,
            norm=matplotlib.colors.LogNorm(),
            shading='nearest')
        plt.plot(
            qy_markers_array,
            qz_markers_array,
            marker='o',
            linestyle='',
            markersize=8,
            markerfacecolor='none',
            markeredgecolor='r',
            markeredgewidth=1,
            label='Open dots')
        plt.xlabel(XLABEL_DICT['qy'])
        plt.ylabel(YLABEL_DICT['qz'])
        plt.xlim(np.min(qy_array[i]), np.max(qy_array[i]))
        plt.ylim(np.min(qz_array[i]), np.max(qz_array[i]))
        plt.colorbar(label='I (a.u.)')
        plt.gca().set_aspect('equal', adjustable='box')

        if crop:
            plt.xlim(
                -qy_array[i][np.where(-qz_array[i] == np.min(-qz_array[i]))],
                -qy_array[i][np.where(-qz_array[i] == np.max(-qz_array[i]))])
            plt.ylim(
                -qz_array[i][np.where(-qy_array[i] == np.max(-qy_array[i]))],
                -qz_array[i][np.where(-qy_array[i] == np.min(-qy_array[i]))])
        plt.show()


def plot_2d_onlymarkers(
        qy_markers_array: np.ndarray,
        qz_markers_array: np.ndarray,
        **kwargs) -> None:
    """Plot 2D map with only marker positions.

    This function takes qy_markers_array and qz_markers_array as inputs and
    creates a 2D plot only showing the markers .

    Args:
        - qy_markers_array (np.ndarray): 2D array of qy marker positions.
        - qz_markers_array (np.ndarray): 2D array of qz marker positions.
            The first index is the serial number of measurement.
        - kwargs:
            - xmin (float): the minimum value of horizontal axis.
            - xmax (float): the maximum value of horizontal axis.
            - ymin (float): the minimum value of vertical axis.
            - ymax (float): the maximum value of vertical axis.
            - alpha (float): the incidence angle in the unit of degrees.
            - phi (float): the in-plane rotation angle in the unit of degrees.
            - write (bool, optional): decide if the plot should be saved
                If not provided, default to False
            - current_index (int): the serial number of the plot to be saved
                If not provided, default to None

    Returns:
        - None
    """
    utils.validate_array_dimension(qy_markers_array, 2)
    utils.validate_array_shape(qz_markers_array, qy_markers_array)
    utils.validate_kwargs(
        {'xmin', 'xmax', 'ymin', 'ymax', 'alpha', 'phi', 'write',
         'current_index'},
        kwargs)

    xmin = kwargs.get('xmin')
    xmax = kwargs.get('xmax')
    ymin = kwargs.get('ymin')
    ymax = kwargs.get('ymax')
    alpha = kwargs.get('alpha')
    phi = kwargs.get('phi')
    write = kwargs.get('write', False)
    current_index = kwargs.get('current_index', None)

    plot_set()
    plt.figure()
    plt.plot(
        qy_markers_array,
        qz_markers_array,
        marker='o',
        linestyle='',
        markersize=8,
        markerfacecolor='none',
        markeredgecolor='r',
        markeredgewidth=1,
        label='Open dots')
    plt.xlabel(XLABEL_DICT['qy'])
    plt.ylabel(YLABEL_DICT['qz'])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(r'$\alpha_i\ =\ {:.2f}°\ \phi\ =\ {:.1f}°$'.format(alpha, phi))
    if not write:
        plt.show()
    else:
        plt.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.25)
        plt.savefig(f'{current_index:d}.png')


def plot_2d_polar(
        azimuth_array: np.ndarray,
        qx_array: np.ndarray,
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        params: list[dict],
        **kwargs) -> None:
    """Plot 2D scattering data after polar transformation.

    Args:
        - azimuth_array (np.ndarray): 3D array of azimuthal angles.
        - qx_array (np.ndarray): 3D array of qx values (in Å^-1 units).
        - qy_array (np.ndarray): 3D array of qy values (in Å^-1 units).
        - qz_array (np.ndarray): 3D array of qz values (in Å^-1 units).
        - images (np.ndarray): 3D array of scattering intensities.
            The first index is the serial number of measurement.
        - params (list[dict]): Each dict contains parameters of a measurement.
            Each dictionary must contain the following keys with string values:
                - 'beamcenter_actual': Beam center position '[y z]' in pixel.
        - kwargs:
            - index_list (list[int], optional): list of indexes to plot.
                If not provided, defaults to [0].

    Returns:
        - None
    """
    utils.validate_array_dimension(images, 3)
    utils.validate_array_shape(
        azimuth_array, qx_array, qy_array, qz_array, images)
    utils.validate_kwargs({'index_list'}, kwargs)

    index_list = kwargs.get('index_list', [0])
    q_array = np.sqrt(qx_array**2 + qy_array**2 + qz_array**2)

    plot_set()
    for i in index_list:
        azimuth = np.degrees(azimuth_array[i])
        plt.figure()
        # The polar transformation is plotted by four quadrants individually
        # to avoid the discontinuity at the boundary of the quadrants creating
        # fake lines in pcolormesh.
        beamcenter_y, beamcenter_z = map(
            float, params[i]['beamcenter_actual'].strip('[]').split())
        print(beamcenter_y, beamcenter_z)
        beamcenter_y = int(np.ceil(beamcenter_y))
        beamcenter_z = int(np.rint(beamcenter_z))
        print(beamcenter_y, beamcenter_z)
        plt.pcolormesh(q_array[i, 0:beamcenter_y, 0:beamcenter_z],
                       azimuth[0:beamcenter_y, 0:beamcenter_z],
                       images[i, 0:beamcenter_y, 0:beamcenter_z],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(q_array[i, 0:beamcenter_y, beamcenter_z:],
                       azimuth[0:beamcenter_y, beamcenter_z:],
                       images[i, 0:beamcenter_y, beamcenter_z:],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(q_array[i, beamcenter_y:, 0:beamcenter_z],
                       azimuth[beamcenter_y:, 0:beamcenter_z],
                       images[i, beamcenter_y:, 0:beamcenter_z],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(q_array[i, beamcenter_y:, beamcenter_z:],
                       azimuth[beamcenter_y:, beamcenter_z:],
                       images[i, beamcenter_y:, beamcenter_z:],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.xlabel(XLABEL_DICT['q'])
        plt.ylabel(YLABEL_DICT['azimuth'])
        plt.colorbar(label='I (a.u.)')
        plt.xlim(np.min(q_array[i]), np.max(q_array[i]))
        plt.ylim(np.min(azimuth), np.max(azimuth))
        plt.yticks([0, 60, 120, 180, 240, 300, 360])
        plt.gca().set_aspect('auto', adjustable='box')
        plt.show()


def plot_2d_gi(
        qx_array: np.ndarray,
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        **kwargs) -> None:
    """Plot 2D scattering date with q_parallel and q_z as the axis with input
    qx, qy, qz, and images array.

    Args:
        - qx_array (np.ndarray): 3D array of qx values (in Å^-1 units).
        - qy_array (np.ndarray): 3D array of qy values (in Å^-1 units).
        - qz_array (np.ndarray): 3D array of qz values (in Å^-1 units).
        - images (np.ndarray): 3D array of scattering intensities.
            The first index is the serial number of measurement.
        - kwargs:
            - index_list (list[int], optional): List of indexes to plot.
                If not provided, defaults to [0].

    Returns:
        - None
    """
    utils.validate_array_dimension(images, 3)
    utils.validate_array_shape(qx_array, qy_array, qz_array, images)
    utils.validate_kwargs({'index_list'}, kwargs)

    index_list = kwargs.get('index_list', [0])

    q_parallel = np.sqrt(qx_array**2 + qy_array**2)
    plot_set()
    for i in index_list:
        plt.figure()
        plt.pcolormesh(q_parallel[i],
                       qz_array[i],
                       images[i] * (qy_array[i] > 0),
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='gourand')
        plt.pcolormesh(-q_parallel[i],
                       qz_array[i],
                       images[i] * (qy_array[i] <= 0),
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.xlim(np.min(np.concatenate((q_parallel[qy_array < 0] * -1,
                                        q_parallel[qy_array >= 0]))),
                 np.max(np.concatenate((q_parallel[qy_array < 0] * -1,
                                        q_parallel[qy_array >= 0]))))
        plt.xlabel(XLABEL_DICT['q_parallel'])
        plt.ylabel(YLABEL_DICT['q_vertical'])
        plt.colorbar(label='I (a.u.)')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def plot_1d(
        scattering_vec: np.ndarray,
        intensity: np.ndarray,
        **kwargs) -> None:
    """Plot 1D plot of intensity as a function of scatterin vector q.

    Args:
        - scattering_vec (np.ndarray): 1D Array of q values.
        - intensity (np.ndarray): 1D Array of intensity values.
        - kwargs:
            - xscale (str, optional): the type of x axis scale.
                If not provided, default to 'linear'
            - yscale (str, optional): the type of y axis scale.
                If not provided, default to 'log'
            - xlabel (str, optional): the label of x axis.
                If not provided, default to 'q'
            - ylabel (str, optional): the label of y axis.
                If not provided, default to 'a.u.'.
            - yticks (list[float], optional): the y label ticks.
                If not provided, default to None

    Returns:
        - None
    """
    utils.validate_array_dimension(scattering_vec, 1)
    utils.validate_array_shape(intensity, scattering_vec)
    utils.validate_kwargs(
        {'xscale', 'yscale', 'xlabel', 'ylabel', 'yticks'}, kwargs)

    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'log')
    xlabel = kwargs.get('xlabel', 'q')
    ylabel = kwargs.get('ylabel', 'a.u.')
    yticks = kwargs.get('yticks', None)

    plot_set()
    plt.figure()
    plt.plot(scattering_vec, intensity)
    plt.xlabel(XLABEL_DICT[xlabel])
    plt.ylabel(YLABEL_DICT[ylabel])
    plt.xscale(xscale)
    plt.yscale(yscale)
    if yticks is not None:
        plt.yticks(yticks)
#    plt.xticks([0.00, 0.05, 0.10, 0.15, 0.20])
#    plt.savefig('1d.png', dpi=400, bbox_inches='tight')
    plt.show()


def plot_1d_compare(
        scattering_vec_1: np.ndarray,
        intensity_1: np.ndarray,
        scattering_vec_2: np.ndarray,
        intensity_2: np.ndarray,
        **kwargs) -> None:
    """
    Plot 1D plot comparing datasets 1 and datasets 2.

    Args:
        - scattering_vec_1 (np.ndarray): 1D array of q values.
        - intensity_1 (np.ndarray): 1D array of intensity values.
        - scattering_vec_2 (np.ndarray): 1D array of q values.
        - intensity_2 (np.ndarray): 1D array of intensity values.
        - kwargs:
            - xscale (str, optional): the type of x axis scale.
                If not provided, default to 'linear'
            - yscale (str, optional): the type of y axis scale.
                If not provided, default to 'log'
            - xlabel (str, optional): the label of x axis.
                If not provided, default to 'q'
            - ylabel (str, optional): the label of y axis.
                If not provided, default to 'a.u.'.
            - yticks (list[float], optional): the y label ticks.
                If not provided, default to None
            - legend (list[str], optional): the legend name list.
                If not provided, default to None
            - legend_fontsize (int, optional): the font size of legend.
                If not provided, default to 20
    Returns:
        - None
    """
    utils.validate_array_dimension(scattering_vec_1, 1)
    utils.validate_array_shape(scattering_vec_1, intensity_1)
    utils.validate_array_dimension(scattering_vec_2, 1)
    utils.validate_array_shape(scattering_vec_2, intensity_2)
    utils.validate_kwargs(
        {'xscale', 'yscale', 'xlabel', 'ylabel', 'yticks', 'legend',
         'legend_fontsize'}, kwargs)

    xscale = kwargs.get('xscale', 'log')
    yscale = kwargs.get('yscale', 'log')
    xlabel = kwargs.get('xlabel', 'q')
    ylabel = kwargs.get('ylabel', 'a.u.')
    yticks = kwargs.get('yticks', None)
    legend = kwargs.get('legend', None)
    legend_fontsize = kwargs.get('legend_fontsize', 20)

    plot_set()
    plt.figure()

    def mouse_move(event):
        x, y = event.xdata, event.ydata
        print(x, y)

    plt.connect('motion_notify_event', mouse_move)
    plt.plot(scattering_vec_1, intensity_1)
    plt.plot(scattering_vec_2, intensity_2)
    plt.xlabel(XLABEL_DICT[xlabel])
    plt.ylabel(YLABEL_DICT[ylabel])
    plt.xscale(xscale)
    plt.yscale(yscale)
    if yticks is not None:
        plt.yticks(yticks)
    if legend is not None:
        plt.legend(legend, fontsize=legend_fontsize)
    plt.show()


# This function has some magic constants that need to be fixed.
def plot_3d_q(
        qx_array: np.ndarray,
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        **kwargs) -> None:
    """Render a three dimension plot with qx, qy, qz array and image array.

    Args:
        - qx_array (np.ndarray): Array of q values in the x direction.
        - qy_array (np.ndarray): Array of q values in the y direction.
        - qz_array (np.ndarray): Array of q values in the z direction.
        - images (np.ndarray): Array of the intensity of scattering images.
        - kwargs:
            - index_list (list[int], optional): List of indices to plot.
                If not provided, defaults to None.

    Returns:
        - None
    """
    utils.validate_array_dimension(qx_array, 3)
    utils.validate_array_shape(qx_array, qy_array, qz_array, images)
    utils.validate_kwargs({'index_list'}, kwargs)

    index_list = kwargs.get('index_list', None)

    plot_set_small()
    for i in index_list:
        color_dimension = images[i]
        minn = np.min(color_dimension[color_dimension > 0])
        maxx = np.max(color_dimension)
        norm = LogNorm(vmin=minn, vmax=maxx)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array([])
        fcolors = mappable.to_rgba(color_dimension)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x_min, x_max = 0, -6  # np.min(qx_array[i]), np.max(qx_array[i])
        y_min, y_max = np.min(qy_array), np.max(qy_array)
        z_min, z_max = np.min(qz_array), np.max(qz_array)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        # Calculate the ranges for each axis
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        # Determine the largest range among the three axes
        max_range = max(x_range, y_range, z_range)
        # Normalize the aspect ratios using the largest range
        aspect_x = x_range / max_range
        aspect_y = y_range / max_range
        aspect_z = z_range / max_range

        # Set the aspect ratio for the x, y, and z axes
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
        # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        # ax.set_xlabel(r'$q_x\ (A^{-1})$')
        # ax.set_ylabel(r'$q_y\ (A^{-1})$')
        # ax.set_zlabel(r'$q_z\ (A^{-1})$')
        ax.set_xticks([-6, -5, -4, -3, -2, -1, 0])
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_zticks([-1, 0, 1])

        ax.plot_surface(
            qx_array[i],
            qy_array[i],
            qz_array[i],
            rstride=1,
            cstride=1,
            facecolors=fcolors,
            vmin=minn,
            vmax=maxx,
            shade=False)
        ax.quiver(-4 * np.pi / 1.542, 0, 0, 2 * np.pi / 1.542, 0, 0,
                  length=1.0, color='k', arrow_length_ratio=0.1)
        ax.text(-5.5, 0, -0.5, r'$k_i$', color='k')
        ax.scatter(-2 * np.pi / 1.542, 0, 0, color='r', s=50)
        ax.text(-2 * np.pi / 1.542, 0, 0.2, 'sample', color='r')

        ax.set_xlabel(XLABEL_DICT['qx'])
        ax.xaxis.labelpad = 15
        ax.set_ylabel(XLABEL_DICT['qy'])
        ax.set_zlabel(XLABEL_DICT['qz'])
        ax.zaxis.labelpad = -30

        ax.grid(False)
        plt.show()


# This function has some magic constants that need to be fixed.
def plot_3d_mm(x_array: np.ndarray,
               y_array: np.ndarray,
               z_array: np.ndarray,
               images: np.ndarray,
               **kwargs) -> None:
    """Render a real-space 3d plot with input x, y, z array and image array.

    Args:
        - x_array (np.ndarray): Array of x of Cartesian coordinates.
        - y_array (np.ndarray): Array of y of Cartesian coordinates.
        - z_array (np.ndarray): Array of z of Cartesian coordinates.
        - images (np.ndarray): Array of the intensity of scattering images.
        - kwargs:
            - index_list (list[int], optional): List of indexes to plot.
                If not provided, defaults to None.

    Returns:
        - None
    """
    utils.validate_array_dimension(x_array, 3)
    utils.validate_array_shape(x_array, y_array, z_array, images)
    utils.validate_kwargs({'index_list'}, kwargs)

    index_list = kwargs.get('index_list', None)

    plot_set_small()
    for i in index_list:
        color_dimension = images[i]
        minn = np.min(color_dimension[color_dimension > 0])
        maxx = np.max(color_dimension)
        norm = LogNorm(vmin=minn, vmax=maxx)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array([])
        fcolors = mappable.to_rgba(color_dimension)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x_min, x_max = 0, np.max(x_array)
        y_min, y_max = np.min(y_array), np.max(y_array)
        z_min, z_max = np.min(z_array), np.max(z_array)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        # Calculate the ranges for each axis
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        # Determine the largest range among the three axes
        max_range = max(x_range, y_range, z_range)
        # Normalize the aspect ratios using the largest range
        aspect_x = x_range / max_range / 10
        aspect_y = y_range / max_range
        aspect_z = z_range / max_range

        # Set the aspect ratio for the x, y, and z axes
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
        # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        # ax.set_xlabel(r'$q_x\ (A^{-1})$')
        # ax.set_ylabel(r'$q_y\ (A^{-1})$')
        # ax.set_zlabel(r'$q_z\ (A^{-1})$')
        # ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_yticks([-30, 0, 30])
        ax.set_zticks([-30, 0, 30])
        ax.invert_xaxis()
        ax.plot_surface(
            x_array[i],
            y_array[i],
            z_array[i],
            rstride=1,
            cstride=1,
            facecolors=fcolors,
            vmin=minn,
            vmax=maxx,
            shade=False)
        # ax.quiver(-100, 0, 0, 100, 0, 0,
        #           length=1.0, color='k', arrow_length_ratio=0.1)
        # ax.text(-50, 0, 10, r'$x-ray$', color='k')
        # ax.scatter(0, 0, 0, color='r', s=20)
        # ax.text(270, 50, 50, 'sample', color='r')

        ax.set_xlabel(r'$x\ \mathrm{(mm)}$')
        ax.xaxis.labelpad = 10
        ax.set_ylabel(r'$y\ \mathrm{(mm)}$')
        ax.yaxis.labelpad = 10
        ax.set_zlabel(r'$z\ \mathrm{(mm)}$')
        ax.zaxis.labelpad = -25

        ax.grid(False)
        plt.show()


# This function is still under development.
def plot_3d_grating(
        qx_array: np.ndarray,
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        images: np.ndarray,
        **kwargs) -> None:
    """Render a 3d grating plot with input qx, qy, qz array and image array.

    Args:
        - qx_array (np.ndarray): 3D array of q values in the x direction.
        - qy_array (np.ndarray): 3D array of q values in the y direction.
        - qz_array (np.ndarray): 3D array of q values in the z direction.
        - images (np.ndarray): 3D array of intensity of scattering images.
        - kwargs:
            - qx (np.ndarray): 2D array of qx values of the crossing point of
                reciprocal lattice vector and the Ewald sphere. The first
                index is the serial number of measurement and the second index
                is the reflection order, in the unit of Å^-1.
            - qy (np.ndarray): 2D array of qy values.
            - qz (np.ndarray): 2D array of qz values.
            - qx_0 (np.ndarray): 2D array of qx_0 values, the crossing point of
                the reciprocal lattice vector and smaple plane in real space.
            - qy_0 (np.ndarray): 2D array of qy_0 values.
            - qz_0 (np.ndarray): 2D array of qz_0 values.
            - index_list (list[int], optional): List of measurements to plot.
                If not provided, defaults to [0].

    Returns:
        - None
    """
    utils.validate_array_dimension(qx_array, 3)
    utils.validate_array_shape(qx_array, qy_array, qz_array, images)
    utils.validate_kwargs({'qx', 'qy', 'qz', 'qx_0', 'qy_0', 'qz_0',
                           'index_list'}, kwargs)

    qx = kwargs['qx']
    qy = kwargs['qy']
    qz = kwargs['qz']
    qx_0 = kwargs['qx_0']
    qy_0 = kwargs['qy_0']
    qz_0 = kwargs['qz_0']
    index_list = kwargs.get('index_list', [0])

    utils.validate_array_dimension(qx, 2)
    utils.validate_array_shape(qx, qy, qz, qx_0, qy_0, qz_0)

    x_min, x_max = np.min(qx_array), np.max(qx_array)
    y_min, y_max = np.min(qy_array), np.max(qy_array)
    z_min, z_max = np.min(qz_array), np.max(qz_array)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    aspect_x = x_range / max_range
    aspect_y = y_range / max_range
    aspect_z = z_range / max_range

    plot_set_small()
    for i in index_list:
        color_dimension = images[i]  # change to desired fourth dimension
        minn, maxx = color_dimension[color_dimension >
                                     0].min(), color_dimension.max()
        norm = LogNorm(vmin=minn, vmax=maxx)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array([])
        fcolors = mappable.to_rgba(color_dimension)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])

        ax.set_xticks([])
        ax.set_yticks([-0.1, 0, 0.1])
        ax.set_zticks([0, 0.05, 0.1, 0.15])

        ax.set_ylabel(XLABEL_DICT['qy'])
        ax.yaxis.labelpad = 30
        ax.set_zlabel(XLABEL_DICT['qz'])
        ax.zaxis.labelpad = 60
        ax.tick_params(axis='z', pad=20)
        ax.tick_params(axis='y', pad=5)
        ax.tick_params(axis='x', pad=0)
        ax.grid(False)

        # projectional relationship cannot be solved in matplotlab
        # use mayavi instead
        for j in range(qx.shape[1]):
            ax.plot((qx_0[i][j], qx[i][j]), (qy_0[i][j], qy[i][j]),
                    (qz_0[i][j], qz[i][j]), color='k')
        ax.plot_surface(
            qx_array[i],
            qy_array[i],
            qz_array[i],
            rstride=1,
            cstride=1,
            facecolors=fcolors,
            vmin=minn,
            vmax=maxx,
            shade=False)
        plt.show()


def plot_2d_columns(q_y: np.ndarray,
                    q_z: np.ndarray,
                    intensity: np.ndarray,
                    **kwargs) -> None:
    """Plot 2D SANS contour plot with input qy, qz array and intensity array.

    Args:
        - q_y (np.ndarray): 1D array of q values in the y direction.
        - q_z (np.ndarray): 1D array of q values in the z direction.
        - intensity (np.ndarray): 1D array of scattering intensity.
        - kwargs:
            - shape (tuple[int]): Shape of the intensity mesh.

    Returns:
        - None
    """
    utils.validate_array_dimension(q_y, 1)
    utils.validate_array_shape(q_y, q_z, intensity)
    utils.validate_kwargs({'shape'}, kwargs)
    shape = kwargs['shape']

    plot_set()
    plt.figure()
    minimum = np.min(np.log10(intensity[intensity > 0]))
    maximum = np.max(np.log10(intensity[intensity > 0]))
    plt.contourf(q_z.reshape(80, 80),
                 -q_y.reshape(80, 80),
                 intensity.reshape(80, 80),
                 cmap='jet',
                 linewidths=3,
                 locator=ticker.LogLocator(),
                 levels=10**np.linspace(minimum, maximum, 400))
    plt.xlabel(XLABEL_DICT['qy'])
    plt.ylabel(XLABEL_DICT['qz'])
    ticks = [10**i for i in range(-10, 20, 1)]
    plt.colorbar(label='I (a.u.)', ticks=ticks)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_1d_sdepth(qz_1d: np.ndarray,
                   depth_1d: np.ndarray,
                   **kwargs) -> None:
    """Plot 1D scattering depth as a function of  vertical scattering vector.

    Args:
        - qz_1d (np.ndarray): 1D array of verticle scattering vector.
        - depth_1d (np.ndarray): 1D array of scattering depth.
        - kwargs:
            - x_max (float, optional): maximun value of x axis.
                If not provided, default to None
            - y_min (float, optional): minimum value of y axis.
                If not provided, default to None
            - y_max (float, optional): maximun value of y axis.
                If not provided, default to None
            - legend (list[str], optional): the legend name list.
                If not provided, default to None

    Returns:
        - None
    """
    utils.validate_array_dimension(qz_1d, 1)
    utils.validate_array_shape(depth_1d, qz_1d)
    utils.validate_kwargs({'x_max', 'y_min', 'y_max', 'legend'}, kwargs)

    x_max = kwargs.get('x_max', None)
    y_min = kwargs.get('y_min', None)
    y_max = kwargs.get('y_max', None)
    legend = kwargs.get('legend', None)
    plot_set()
    plt.plot(qz_1d, depth_1d)
    plt.yscale('log')
    if x_max is not None:
        plt.xlim(0, x_max)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.ylabel(YLABEL_DICT['depth'])
    plt.xlabel(XLABEL_DICT['qz0'])
    if legend is not None:
        plt.legend(legend, fontsize=19)
    plt.show()


def plot_1d_sdepth_compare(qz1_1d: np.ndarray,
                           depth1_1d: np.ndarray,
                           qz2_1d: np.ndarray,
                           depth2_1d: np.ndarray,
                           **kwargs) -> None:
    """Compare 1D scattering depth of dataset 1 and dataset 2.

    Args:
        - qz1_1d (np.ndarray): 1D array of verticle scattering vector.
        - depth1_1d (np.ndarray): 1D array of scattering depth.
        - qz2_1d (np.ndarray): 1D array of verticle scattering vector.
        - depth2_1d (np.ndarray): 1D array of scattering depth.
        - kwargs:
            - x_max (float, optional): maximun value of x axis.
                If not provided, default to None
            - y_min (float, optional): minimum value of y axis.
                If not provided, default to None
            - y_max (float, optional): maximun value of y axis.
                If not provided, default to None
            - legend (list[str], optional): the legend name list.
                If not provided, default to None

    Returns:
        - None
    """
    utils.validate_array_dimension(qz1_1d, 1)
    utils.validate_array_shape(qz1_1d, depth1_1d, qz2_1d, depth2_1d)
    utils.validate_kwargs({'x_max', 'y_min', 'y_max', 'legend'}, kwargs)

    x_max = kwargs.get('x_max', None)
    y_min = kwargs.get('y_min', None)
    y_max = kwargs.get('y_max', None)
    legend = kwargs.get('legend', None)
    plot_set()
    plt.plot(qz1_1d, depth1_1d)
    plt.plot(qz2_1d, depth2_1d)
    plt.yscale('log')
    if x_max is not None:
        plt.xlim(0, x_max)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.ylabel(YLABEL_DICT['depth'])
    plt.xlabel(XLABEL_DICT['qz0'])
    if legend is not None:
        plt.legend(legend, fontsize=19)
    plt.show()


def plot_2d_sdepth(ki_mesh: np.ndarray,
                   kf_mesh: np.ndarray,
                   depth_mesh: np.ndarray,
                   **kwargs) -> None:
    """plot 2D scattering depth as a function of ki and kf.

    Args:
        - ki_mesh (np.ndarray): 2D array of incidence wavevector.
        - kf_mesh (np.ndarray): 2D array of exit wavevector.
        - depth_mesh (np.ndarray): 2D array of scattering depth.
        - kwargs:
            - x_max (float, optional): maximun value of x axis.
                If not provided, default to None
            - y_max (float, optional): maximun value of y axis.
                If not provided, default to None

    Returns:
        - None
    """
    utils.validate_array_dimension(ki_mesh, 2)
    utils.validate_array_shape(ki_mesh, kf_mesh, depth_mesh)
    utils.validate_kwargs({'x_max', 'y_max'}, kwargs)

    x_max = kwargs.get('x_max', None)
    y_max = kwargs.get('y_max', None)
    plot_set()
    plt.pcolormesh(ki_mesh,
                   kf_mesh,
                   depth_mesh,
                   cmap='jet',
                   linewidths=3,
                   norm=matplotlib.colors.LogNorm(),
                   shading='nearest')
    if x_max is not None:
        plt.xlim(0, x_max)
    if y_max is not None:
        plt.ylim(0, y_max)
    plt.xlabel(XLABEL_DICT['kzi'])
    plt.ylabel(YLABEL_DICT['kzf'])
    # plt.xticks([0.00, 0.05, 0.10])
    plt.colorbar().set_label(label=YLABEL_DICT['depth'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
