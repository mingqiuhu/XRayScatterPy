"""
This module provides functions to plot the processed scattering data. 
The main functions in this module are used to:

1. Set the global matplotlib settings with certain parameters (plot_set, plot_set_small).
2. Plot different types of 2D scattering plots with or without pidiction lines in cartesian
   and polar coordinates (plot_2d_scattering, plot_2d_scattering_withlines, 
   plot_2d_scattering_onlylines, plot_2d_polar).
3. Plot 2D scattering plots in q_parallel axis and q_z axis with intensity colormap 
   (plot_2d_paralell).
4. Plot specular reflectivity plots (plot_specular_reflectivity).
5. Plot and compare 1D scattering plots (plot_1d, plot_1d_compare)
6. plot 3D scattering plots in real space with x, y, z axis and intensity colormap
   (plot_3D, plot_3D_mm, plot_3D_grating)
7. plot 2D Small Angle Neutron Scattering plot (plot_sans)
8. plot 1D penentration plot (plot_penetration, plot_penetration_compare)
10. plot 2D penentration plot (plot_penetration_2D)

These functions take in various q arraies and intensity arries calculated and processed by 
other modules like calibrations.py and dataprocessing.py
"""

import numpy as np
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

def plot_set():
    """Update the global matplotlib settings with following parameters for
    higher figure resolution, larger tick labels, and thicker lines

    Args:
        None

    Returns:
        None
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['figure.dpi'] = 600
    matplotlib.rcParams['xtick.labelsize'] = 22
    matplotlib.rcParams['ytick.labelsize'] = 22
    matplotlib.rcParams['font.size'] = 22
    matplotlib.rcParams['legend.fontsize'] = 22
    matplotlib.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['xtick.major.width'] = 3
    matplotlib.rcParams['xtick.minor.width'] = 3
    matplotlib.rcParams['ytick.major.width'] = 3
    matplotlib.rcParams['ytick.minor.width'] = 3


def plot_set_small():
    """Update the global matplotlib settings with following parameters for
    higher figure resolution, medium tick labels, and thicker lines

    Args:
        None

    Returns:
        None
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


def plot_2d_scattering(qy_array: np.ndarray, qz_array: np.ndarray, image_array: np.ndarray,
                       **kwargs):
    """This function takes qy_array, qz_array, and image_array as inputs and
    creates a 2D colormap plot for each index in index_list using the
    provided arrays.

    Args:
        - qy_array (np.ndarray): 1D array of qy values (in A^-1 units).
        - qz_array (np.ndarray): 1D array of qz values (in A^-1 units).
        - image_array (np.ndarray): 2D array of scattering intensities.
        - kwargs:
            - index_list (list[int], optional): list of indices for which to create plots.
                If not provided, defaults to None.
            - crop (bool, optional): decide whether the plot would be cropped.
                If not providded, default to False
            - xticks (list[float], optional): set the xtick locations.
                If not provided, default to None
            - yticks (list[float], optional): set the ytick locations.
                If not provided, default to None
            - video (bool, optional): decide if the images would be used to generate videos
                If not provided, default to False
    Returns:
        None
    """
    index_list = kwargs.get('index_list', None)
    crop = kwargs.get('crop', False)
    xticks = kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    video = kwargs.get('video', False)
    if index_list is None:
        index_list = [0]

    plot_set()
    for i in index_list:
        plt.figure()
        zmax = np.max(image_array[i])
        zmin = np.min(image_array[i][image_array[i] > 0])
        # zmin = max(zmax*1e-6, np.min(image_array[i][image_array[i] > 0]))
        norm = matplotlib.colors.LogNorm(zmin, zmax)
        plt.pcolormesh(qy_array[i],
                       qz_array[i],
                       image_array[i],
                       cmap='jet',
                       linewidths=3,
                       norm=norm,
                       shading='nearest')
        plt.xlabel(r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$', fontsize=22)
        plt.ylabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
        if not video:
            plt.colorbar(label='I (a.u.)')
        plt.gca().set_aspect('equal', adjustable='box')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        if crop:
            plt.xlim(-qy_array[i][np.where(-qz_array[i] == np.min(-qz_array[i]))
                                  ], -qy_array[i][np.where(-qz_array[i] == np.max(-qz_array[i]))])
            plt.ylim(-qz_array[i][np.where(-qy_array[i] == np.max(-qy_array[i]))
                                  ], -qz_array[i][np.where(-qy_array[i] == np.min(-qy_array[i]))])
        if video:
            plt.xlim(np.min(qy_array), np.max(qy_array))
            plt.ylim(np.min(qz_array), np.max(qz_array))
        plt.show()


def plot_2d_scattering_withlines(qy_array: np.ndarray, qz_array: np.ndarray,
                                 image_array: np.ndarray,
                                 qy_lines_array: np.ndarray, qz_lines_array: np.ndarray,
                                 **kwargs):
    """This function takes qy_array, qz_array, and image_array as inputs and
    creates a 2D colormap plot with lines for each index in index_list using the
    provided arrays.

    Args:
        - qy_array (np.ndarray): 1D array of qy values (in A^-1 units).
        - qz_array (np.ndarray): 1D array of qz values (in A^-1 units).
        - image_array (np.ndarray): 2D array of scattering intensities.
        - qy_lines_array (np.ndarray): 1D array of qy lines values.
        - qz_lines_array (np.ndarray): 1D array of qz lines values
        - kwargs:
            - index_list (list[int], optional): list of indices for which to create plots.
                If not provided, defaults to None.
            - crop (bool, optional): decide whether the plot would be cropped.
                If not providded, default to False

    Returns:
        None
    """
    index_list = kwargs.get('index_list', None)
    crop = kwargs.get('crop', False)
    if index_list is None:
        index_list = [0]

    plot_set()
    for i in index_list:
        plt.figure()
        plt.pcolormesh(qy_array[i],
                       qz_array[i],
                       image_array[i],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.plot(qy_lines_array, qz_lines_array, marker='o', linestyle='', markersize=8,
                 markerfacecolor='none', markeredgecolor='r', markeredgewidth=1, label='Open dots')
        plt.xlabel(r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$', fontsize=22)
        plt.ylabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
        plt.xlim(np.min(qy_array[i]), np.max(qy_array[i]))
        plt.ylim(np.min(qz_array[i]), np.max(qz_array[i]))
        plt.colorbar(label='I (a.u.)')
        plt.gca().set_aspect('equal', adjustable='box')

        if crop:
            plt.xlim(-qy_array[i][np.where(-qz_array[i] == np.min(-qz_array[i]))
                                  ], -qy_array[i][np.where(-qz_array[i] == np.max(-qz_array[i]))])
            plt.ylim(-qz_array[i][np.where(-qy_array[i] == np.max(-qy_array[i]))
                                  ], -qz_array[i][np.where(-qy_array[i] == np.min(-qy_array[i]))])

        plt.show()


def plot_2d_scattering_onlylines(qy_lines_array: np.ndarray, qz_lines_array: np.ndarray,
                                 **kwargs):
    """Plot 2D scattering data as a colormap.

    This function takes qy_array, qz_array, and image_array as inputs and
    creates a 2D colormap plot for each index in index_list using the
    provided arrays.

    Args:
        - qy_lines_array (np.ndarray): 1D array of qy lines values.
        - qz_lines_array (np.ndarray): 1D array of qz lines values.
        - kwargs:
            - xmin (float): the minimum value of horizontal axis.
            - xmax (float): the maximum value of horizontal axis.
            - ymin (float): the minimum value of vertical axis.
            - ymax (float): the maximum value of vertical axis.
            - alpha (float, optional): the transparancy of the lines.
                If not provided, default to 0.2
            - phi (float, optional): 
                If not provided, default to -0.04
            - write (bool, optional): decide if the plot should be saved
                If not provided, default to False
            - current_index (int): the index of the plot to be saved
                If not provided, default to None
            
    Returns:
        None
    """

    xmin = kwargs.get('xmin')
    xmax = kwargs.get('xmax')
    ymin = kwargs.get('ymin')
    ymax = kwargs.get('ymax')
    alpha = kwargs.get('alpha', 0.2)
    phi = kwargs.get('phi', -0.04)
    write = kwargs.get('write',  False)
    current_index = kwargs.get('current_index', None)

    plot_set()
    plt.figure()
    plt.plot(qy_lines_array, qz_lines_array, marker='o', linestyle='', markersize=8,
             markerfacecolor='none', markeredgecolor='r', markeredgewidth=1, label='Open dots')
    plt.xlabel(r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$', fontsize=22)
    plt.ylabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(r'$\alpha_i\ =\ {:.2f}°\ \phi\ =\ {:.1f}°$'.format(alpha, phi))
    if not write:
        plt.show()
    else:
        plt.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.25)
        plt.savefig(f'{current_index:d}.png')


def plot_2d_polar(azimuth_array: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray,
                  qz_array: np.ndarray, image_array:np.ndarray, **kwargs):
    """
    Plot 2D polar scattering data.

    Args:
        - azimuth_array (np.ndarray): 1D array of azimuthal angles.
        - qx_array (np.ndarray): 1D array of q values in the x direction.
        - qy_array (np.ndarray): 1D array of q values in the y direction.
        - qz_array (np.ndarray): 1D array of q values in the z direction.
        - image_array (np.ndarray): 1D array of scattering images.
        - kwargs:
            - index_list (list[int], optional): list of indices to plot. 
                If not provided, defaults to None.
    
    Returns:
        None
    """
    index_list = kwargs.get('index_list', None)
    if index_list is None:
        index_list = [0]

    plot_set()
    for i in index_list:
        q = np.sqrt(qx_array[i]**2 + qy_array[i]**2 + qz_array[i]**2)
        azimuth = np.degrees(azimuth_array[i])
        image = image_array[i]
        plt.figure()
        plt.pcolormesh(q[0:351, 0:214],
                       azimuth[0:351, 0:214],
                       image[0:351, 0:214],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(q[0:351, 214:],
                       azimuth[0:351, 214:],
                       image[0:351, 214:],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(q[351:, 0:214],
                       azimuth[351:, 0:214],
                       image[351:, 0:214],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(q[351:, 214:],
                       azimuth[351:, 214:],
                       image[351:, 214:],
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.xlabel(r'$q\ \mathrm{(Å^{-1})}$')
        plt.ylabel(r'$\mathrm{azimuth}\ (°)$')
        plt.colorbar(label='I (a.u.)')
        plt.xlim(np.min(q), np.max(q))
        plt.ylim(np.min(azimuth), np.max(azimuth))
        yticks = [0, 60, 120, 180, 240, 300, 360]
        plt.yticks(yticks)
        plt.gca().set_aspect('auto', adjustable='box')
        plt.show()


def plot_2d_paralell(qx_array: np.ndarray, qy_array:np.ndarray, qz_array: np.ndarray,
                    image_array: np.ndarray, **kwargs):
    """
    Plot 2D plot in q_parallel and q_z axis with input qx, qy, qz array and image array.

    Args:
        - qx_array (np.ndarray): Array of q values in the x direction.
        - qy_array (np.ndarray): Array of q values in the y direction.
        - qz_array (np.ndarray): Array of q values in the z direction.
        - image_array (np.ndarray): Array of scattering images.
        - kwargs:
            - index_list (list[int], optional): List of indices to plot. 
                If not provided, defaults to None.
    
    Returns:
        None
    """
    index_list = kwargs.get('index_list', None)
    if index_list is None:
        index_list = [0]

    q_paralell = np.sqrt(qx_array**2 + qy_array**2)
    plot_set()
    for i in index_list:

        plt.figure()
        plt.pcolormesh(q_paralell[i] * (qy_array[i] > 0),
                       qz_array[i] * (qy_array[i] > 0),
                       image_array[i] * (qy_array[i] > 0),
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.pcolormesh(-q_paralell[i] * (qy_array[i] <= 0),
                       qz_array[i] * (qy_array[i] <= 0),
                       image_array[i] * (qy_array[i] <= 0),
                       cmap='jet',
                       linewidths=3,
                       norm=matplotlib.colors.LogNorm(),
                       shading='nearest')
        plt.xlabel(r'$q_\Vert\ \mathrm{(Å^{-1})}$', fontsize=22)
        plt.ylabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
        plt.colorbar(label='I (a.u.)')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def plot_specular_reflectivity(data: np.ndarray, **kwargs):
    """
    Plot specular reflectivity data.

    Args:
        - data (np.ndarray): The specular reflectivity data array.
        - kwargs:
            - title (str, optional): The title for the plot. 
                If not provided, defaults to 'Specular Reflectivity'.
    
    Returns:
        None
    """
    title = kwargs.get('title', 'Specular Reflectivity')
    plt.figure()
    plt.plot(data)
    plt.xlabel('Incidence Angle')
    plt.ylabel('Reflectivity')
    plt.title(title)
    plt.show()


def plot_1d(q: np.ndarray, i: np.ndarray, **kwargs):
    """
    Plot 1D plot of intensity to q.

    Args:
        - q (np.ndarray): 1D Array of q values.
        - i (np.ndarray): 1D Array of intensity values.
        - kwargs:
            - xscale (str, optional): the type of x axis scale.
                If not provided, default to 'linear'
            - yscale (str, optional): the type of y axis scale.
                If not provided, default to 'log'
            - xlabel (str, optional): the label of x axis.
                If not provided, default to 'qz'
            - ylabel (str, optional): the label of y axis.
                If not provided, default to None
            - yunit (str, optional): the unit of y axis.
                If not provided, default to 'a.u.'
            - yticks (list[float], optional): the y label ticks.
                If not provided, default to None

    Returns:
        None
    """
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'log')
    xlabel = kwargs.get('xlabel', 'qz')
    ylabel = kwargs.get('ylabel', None)
    yunit = kwargs.get('yunit', 'a.u.')
    yticks = kwargs.get('yticks', None)
    plot_set()
    plt.figure()
    plt.plot(q, i)
    if xlabel == 'q':
        plt.xlabel(r'$q\ \mathrm{(Å^{-1})}$', fontsize=22)
    elif xlabel == 'qz':
        plt.xlabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
    elif xlabel == 'theta sample':
        plt.xlabel(r'${\theta}_\mathrm{sample}\ (°)$', fontsize=22)
    if yunit == 'abs':
        plt.ylabel(r'$I\ \mathrm{(cm^{-1}sr^{-1})}$', fontsize=22)
    elif yunit == 'a.u.':
        plt.ylabel(r'$I\ \mathrm{(a.u.)}$', fontsize=22)
    elif yunit == 'normalized reflectivity':
        plt.ylabel(r'$R$', fontsize=22)
    if ylabel == 'total':
        plt.ylabel(r'$I_\mathrm{total}\ \mathrm{(a.u.)}$', fontsize=22)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if yticks is not None:
        plt.yticks(yticks)
    plt.show()


def plot_1d_compare(q1: np.ndarray, i1: np.ndarray,
                    q2: np.ndarray, i2: np.ndarray,
                    **kwargs):
    """
    Plot 1D plot comparing datasets 1 and datasets 2.

    Args:
        - q1 (np.ndarray): Array of q values in the x direction.
        - i1 (np.ndarray): Array of q values in the y direction.
        - q2 (np.ndarray): Array of q values in the z direction.
        - i2 (np.ndarray): Array of scattering images.
        - kwargs:
            - xscale (str, optional): the type of x axis scale.
                If not provided, default to 'log'
            - yscale (str, optional): the type of y axis scale.
                If not provided, default to 'log'
            - xlabel (str, optional): the label of x axis.
                If not provided, default to 'q'
            - ylabel (str, optional): the label of y axis.
                If not provided, default to None
            - yunit (str, optional): the unit of y axis.
                If not provided, default to 'abs'
            - legend (str, optional): the legend name.
                If not provided, default to None
            - legend_fontsize (int, optional): the font size of legend.
                If not provided, default to 20
    Returns:
        None
    """
    xscale = kwargs.get('xscale', 'log')
    xlabel = kwargs.get('xlabel', 'q')
    ylabel = kwargs.get('ylabel', 'spill over')
    yunit = kwargs.get('yunit', 'abs')
    yscale = kwargs.get('yscale', 'log')
    legend = kwargs.get('legend', None)
    legend_fontsize = kwargs.get('legend_fontsize', 20)

    plot_set()
    plt.figure()
    plt.plot(q1, i1)
    plt.plot(q2, i2)
    if xlabel == 'q':
        plt.xlabel(r'$q\ \mathrm{(Å^{-1})}$', fontsize=22)
    elif xlabel == 'qz':
        plt.xlabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
    elif xlabel == 'theta sample':
        plt.xlabel(r'${\theta}_\mathrm{sample}\ (°)$', fontsize=22)
    if yunit == 'abs':
        plt.ylabel(r'$I\ \mathrm{(cm^{-1}sr^{-1})}$', fontsize=22)
    elif yunit == 'a.u.':
        plt.ylabel(r'$I\ \mathrm{(a.u.)}$', fontsize=22)
    elif yunit == 'normalized reflectivity':
        plt.ylabel(r'$R$', fontsize=22)
    if ylabel == 'spill over':
        plt.ylabel(r'$I_\mathrm{spill\ over}\ \mathrm{(a.u.)}$', fontsize=22)
    if legend is not None:
        plt.legend(legend, fontsize=legend_fontsize)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()


def plot_3d(qx_array: np.ndarray, qy_array: np.ndarray, qz_array: np.ndarray,
            image_array: np.ndarray, **kwargs):
    """
    Plot 3D plot with qx, qy, qz array and image array.

    Args:
        - qx_array (np.ndarray): Array of q values in the x direction.
        - qy_array (np.ndarray): Array of q values in the y direction.
        - qz_array (np.ndarray): Array of q values in the z direction.
        - image_array (np.ndarray): Array of scattering images.
        - kwargs:
            - index_list (list[int], optional): List of indices to plot. 
                If not provided, defaults to None.
    
    Returns:
        None
    """
    index_list = kwargs.get('index_list',None)
    if index_list is None:
        index_list = [0]

    plot_set_small()
    for i in index_list:
        color_dimension = image_array[i]  # change to desired fourth dimension
        minn, maxx = np.min(
            color_dimension[color_dimension > 0]), np.max(color_dimension)
        norm = LogNorm(vmin=minn, vmax=maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x_min, x_max = 0, 6  # np.min(qx_array[i]), np.max(qx_array[i])
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
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_zticks([-1, 0, 1])

        ax.plot_surface(qx_array[i], qy_array[i], qz_array[i], rstride=1,
                        cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
        ax.quiver(4*np.pi/1.542, 0, 0, -2*np.pi/1.542, 0, 0,
                  length=1.0, color='k', arrow_length_ratio=0.1)
        ax.text(5.5, 0, -0.5, r'$k_i$', color='k')
        ax.scatter(2*np.pi/1.542, 0, 0, color='r', s=50)
        ax.text(2*np.pi/1.542, 0, 0.2, 'sample', color='r')

        ax.set_xlabel(r'$q_\mathrm{x}\ \mathrm{(Å^{-1})}$')
        ax.xaxis.labelpad = 15
        ax.set_ylabel(r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$')
        ax.set_zlabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$')
        ax.zaxis.labelpad = -30

        ax.grid(False)
        plt.show()


def plot_3d_mm(x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray,
               image_array: np.ndarray, **kwargs):
    """
    Plot 3D plot with input qx, qy, qz array and image array in smaller tick size.

    Args:
        - qx_array (np.ndarray): Array of q values in the x direction.
        - qy_array (np.ndarray): Array of q values in the y direction.
        - qz_array (np.ndarray): Array of q values in the z direction.
        - image_array (np.ndarray): Array of scattering images.
        - kwargs:
            - index_list (list[int], optional): List of indices to plot. 
                If not provided, defaults to None.
    
    Returns:
        None
    """
    index_list = kwargs.get('index_list', None)
    if index_list is None:
        index_list = [0]

    plot_set_small()
    for i in index_list:
        color_dimension = image_array[i]  # change to desired fourth dimension
        minn, maxx = np.min(
            color_dimension[color_dimension > 0]), np.max(color_dimension)
        norm = LogNorm(vmin=minn, vmax=maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

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
        ax.plot_surface(x_array[i], y_array[i], z_array[i], rstride=1,
                        cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
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

def plot_3d_grating(qx_array: np.ndarray, qy_array: np.ndarray, qz_array: np.ndarray,
                    image_array: np.ndarray, **kwargs):
    """
    Plot 3D grating plot with input qx, qy, qz array and image array.

    Args:
        - qx_array (np.ndarray): Array of q values in the x direction.
        - qy_array (np.ndarray): Array of q values in the y direction.
        - qz_array (np.ndarray): Array of q values in the z direction.
        - image_array (np.ndarray): Array of scattering images.
        - kwargs:
            - qx (np.ndarray): List of qx values. 
            - qy (np.ndarray): List of qy values. 
            - qz (np.ndarray): List of qz values. 
            - qx_0 (np.ndarray): List of qx_0 values. 
            - qy_0 (np.ndarray): List of qy_0 values. 
            - qz_0 (np.ndarray): List of qz_0 values. 
            - index_list (list[int], optional): List of indices to plot. 
                If not provided, defaults to None.
    
    Returns:
        None
    """

    qx = kwargs.get('qx')
    qy = kwargs.get('qy')
    qz = kwargs.get('qz')
    qx_0 = kwargs.get('qx_0')
    qy_0 = kwargs.get('qy_0')
    qz_0 = kwargs.get('qz_0')
    index_list = kwargs.get('index_list', None)
    if index_list is None:
        index_list = [0]

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
        color_dimension = image_array[i]  # change to desired fourth dimension
        minn, maxx = color_dimension[color_dimension >
                                     0].min(), color_dimension.max()
        norm = LogNorm(vmin=minn, vmax=maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

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

        ax.set_ylabel(r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$')
        ax.yaxis.labelpad = 30
        ax.set_zlabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$')
        ax.zaxis.labelpad = 60
        ax.tick_params(axis='z', pad=20)
        ax.tick_params(axis='y', pad=5)
        ax.tick_params(axis='x', pad=0)
        ax.grid(False)

        # this cannot be solved in matplotlab, use mayavi instead
        for j in range(qx.shape[1]):
            ax.plot((qx_0[i][j], qx[i][j]), (qy_0[i][j], qy[i][j]),
                    (qz_0[i][j], qz[i][j]), color='k')
        ax.plot_surface(qx_array[i], qy_array[i], qz_array[i], rstride=1,
                        cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
        plt.show()


def plot_sans(qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray):
    """
    Plot 2D SANS contour plot with input qy, qz array and intensity array.

    Args:
        - qy (np.ndarray): 1D array of q values in the y direction.
        - qz (np.ndarray): 1D array of q values in the z direction.
        - intensity (np.ndarray): 2D array of scattering intensity.
        
    Returns:
        None
    """
    plot_set()
    plt.figure()
    MIN = np.min(np.log10(intensity[intensity > 0]))
    MAX = np.max(np.log10(intensity[intensity > 0]))
    plt.contourf(qz.reshape(80, 80),
                 -qy.reshape(80, 80),
                 intensity.reshape(80, 80),
                 cmap='jet',
                 linewidths=3,
                 locator=ticker.LogLocator(),
                 levels=10**np.linspace(MIN, MAX, 400))
    # ,vmax=10**(MAX-0.5))
    plt.xlabel(r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$', fontsize=22)
    plt.ylabel(r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$', fontsize=22)
    plt.colorbar(label='I (a.u.)', ticks=[
                 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_penetration(qz_1d: np.ndarray, depth_1d: np.ndarray, **kwargs):
    """
    Plot 1D penetration plot with input 1D qz array and 1D depth array.

    Args:
        - qz_1d (np.ndarray): 1D array of qz values.
        - depth_1d (np.ndarray): 1D array of depth information.
        - kwargs:
            - x_max (float, optional): maximun value of x axis.
                If not provided, default to None
            - y_min (float, optional): minimum value of y axis.
                If not provided, default to None
            - y_max (float, optional): maximun value of y axis.
                If not provided, default to None
            - legend (str, optional): name of the legend.
                If not provided, default to None
        
    Returns:
        None
    """
    x_max=kwargs.get('x_max', None)
    y_min=kwargs.get('y_min', None)
    y_max=kwargs.get('y_max', None)
    legend = kwargs.get('legend', None)
    plot_set()
    plt.plot(qz_1d, depth_1d)
    plt.yscale('log')
    if x_max is not None:
        plt.xlim(0, x_max)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.ylabel(r'$z_\mathrm{1/e}\ \mathrm{(Å)}$')
    plt.xlabel(r'$q_\mathrm{z,0}\ \mathrm{(Å^{-1})}$')
    if legend is not None:
        plt.legend(legend, fontsize=19)
    plt.show()


def plot_penetration_compare(qz1_1d: np.ndarray, depth1_1d: np.ndarray,
                             qz2_1d: np.ndarray, depth2_1d: np.ndarray,
                             **kwargs):
    """
    Compare 1D penetration plot with of dataset 1 and dataset 2.

    Args:
        - qz1_1d (np.ndarray): 1D array of qz1 values.
        - depth1_1d (np.ndarray): 1D array of depth1 information.
        - qz2_1d (np.ndarray): 1D array of qz2 values.
        - depth2_1d (np.ndarray): 1D array of depth2 information.
        - kwargs:
            - x_max (float, optional): maximun value of x axis.
                If not provided, default to None
            - y_min (float, optional): minimum value of y axis.
                If not provided, default to None
            - y_max (float, optional): maximun value of y axis.
                If not provided, default to None
            - legend (str, optional): name of the legend.
                If not provided, default to None
        
    Returns:
        None
    """
    x_max=kwargs.get('x_max', None)
    y_min=kwargs.get('y_min', None)
    y_max=kwargs.get('y_max', None)
    legend = kwargs.get('legend', None)
    plot_set()
    plt.plot(qz1_1d, depth1_1d)
    plt.plot(qz2_1d, depth2_1d)
    plt.yscale('log')
    if x_max is not None:
        plt.xlim(0, x_max)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.ylabel(r'$z_\mathrm{1/e}\ \mathrm{(Å)}$')
    plt.xlabel(r'$q_\mathrm{z,0}\ \mathrm{(Å^{-1})}$')
    plt.legend(legend, fontsize=19)
    plt.show()


def plot_penetration_2d(ki_mesh: np.ndarray, kf_mesh: np.ndarray,
                        depth_mesh: np.ndarray, **kwargs):
    """
    plot 2D penetration plot ki, kf and depth mesh array.

    Args:
        - ki_mesh (np.ndarray): Array of ki values.
        - kf_mesh (np.ndarray): Array of kf values.
        - depth_mesh (np.ndarray): Array of depth information.
        - kwargs:
            - x_max (float, optional): maximun value of x axis.
                If not provided, default to None
            - y_max (float, optional): maximun value of y axis.
                If not provided, default to None
        
    Returns:
        None
    """
    x_max=kwargs.get('x_max', None)
    y_max=kwargs.get('y_max', None)
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
    plt.xlabel(r'$k_\mathrm{z,i}\ \mathrm{(Å^{-1})}$',
               fontsize=22, fontstyle='normal')
    plt.ylabel(r'$k_\mathrm{z,f}\ \mathrm{(Å^{-1})}$',
               fontsize=22, fontstyle='normal')
    # plt.xticks([0.00, 0.05, 0.10])
    plt.colorbar().set_label(label=r'$z_\mathrm{1/e}\ \mathrm{(Å)}$', size=22)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.rcParams.update({'font.size': 22})
    plt.rcParams.update({'legend.fontsize': 22})
    plt.show()
