# xray_scatter_py/data_plotting.py

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import ticker


def plot_set():
    """Update the global matplotlib settings for the current session.

    This function modifies the default matplotlib settings for better
    visualizations, such as higher DPI, larger tick labels, and thicker lines.

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
    """Update the global matplotlib settings for the current session.

    This function modifies the default matplotlib settings for better
    visualizations, such as higher DPI, larger tick labels, and thicker lines.

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


def plot_2d_scattering(qy_array, qz_array, image_array, index_list=None, crop=False, XTICKS=None, YTICKS=None, video=False):
    """Plot 2D scattering data as a colormap.

    This function takes qy_array, qz_array, and image_array as inputs and
    creates a 2D colormap plot for each index in index_list using the
    provided arrays.

    Args:
        qy_array (np.ndarray): 1D array of qy values (in A^-1 units).
        qz_array (np.ndarray): 1D array of qz values (in A^-1 units).
        image_array (np.ndarray): 2D array of scattering intensities.
        index_list (list, optional): List of indices for which to create plots.
                                     Defaults to [0].

    Returns:
        None
    """
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
        if XTICKS is not None:
            plt.xticks(XTICKS)
        if YTICKS is not None:
            plt.yticks(YTICKS)
        if crop:
            plt.xlim(-qy_array[i][np.where(-qz_array[i] == np.min(-qz_array[i]))
                                  ], -qy_array[i][np.where(-qz_array[i] == np.max(-qz_array[i]))])
            plt.ylim(-qz_array[i][np.where(-qy_array[i] == np.max(-qy_array[i]))
                                  ], -qz_array[i][np.where(-qy_array[i] == np.min(-qy_array[i]))])
        if video:
            plt.xlim(np.min(qy_array), np.max(qy_array))
            plt.ylim(np.min(qz_array), np.max(qz_array))
        plt.show()


def plot_2d_scattering_withlines(qy_array, qz_array, image_array, qy_lines_array, qz_lines_array, index_list=None, crop=False):
    """Plot 2D scattering data as a colormap.

    This function takes qy_array, qz_array, and image_array as inputs and
    creates a 2D colormap plot for each index in index_list using the
    provided arrays.

    Args:
        qy_array (np.ndarray): 1D array of qy values (in A^-1 units).
        qz_array (np.ndarray): 1D array of qz values (in A^-1 units).
        image_array (np.ndarray): 2D array of scattering intensities.
        index_list (list, optional): List of indices for which to create plots.
                                     Defaults to [0].

    Returns:
        None
    """
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


def plot_2d_scattering_onlylines(qy_lines_array, qz_lines_array, alpha, phi, xmin, xmax, ymin, ymax, write=False, current_index=None):
    """Plot 2D scattering data as a colormap.

    This function takes qy_array, qz_array, and image_array as inputs and
    creates a 2D colormap plot for each index in index_list using the
    provided arrays.

    Args:
        qy_array (np.ndarray): 1D array of qy values (in A^-1 units).
        qz_array (np.ndarray): 1D array of qz values (in A^-1 units).
        image_array (np.ndarray): 2D array of scattering intensities.
        index_list (list, optional): List of indices for which to create plots.
                                     Defaults to [0].

    Returns:
        None
    """
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


def plot_2d_polar(azimuth_array, qx_array, qy_array, qz_array, image_array, index_list=None):
    """
    Plot 2D polar scattering data.

    Args:
        azimuth_array (numpy.ndarray): Array of azimuthal angles.
        qx_array (numpy.ndarray): Array of q values in the x direction.
        qy_array (numpy.ndarray): Array of q values in the y direction.
        qz_array (numpy.ndarray): Array of q values in the z direction.
        image_array (numpy.ndarray): Array of scattering images.
        index_list (list, optional): List of indices to plot. Defaults to None.
    """
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


def plot_specular_reflectivity(data, title="Specular Reflectivity"):
    """
    Plot specular reflectivity data.

    Args:
        data (numpy.ndarray): The specular reflectivity data.
        title (str, optional): The title for the plot. Defaults to "Specular Reflectivity".
    """
    plt.figure()
    plt.plot(data)
    plt.xlabel('Incidence Angle')
    plt.ylabel('Reflectivity')
    plt.title(title)
    plt.show()


def plot_2d_paralell(qx_array, qy_array, qz_array, image_array, index_list=None):

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


def plot_1d_compare(q1, i1, q2, i2, xscale='log', xlabel='q', ylabel=None, yunit='abs',
                    yscale='log', legend=None, legend_fontsize=20):
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


def plot_1d(q, i, xscale='linear', xlabel='qz', yunit='a.u.', ylabel=None, yscale='log', yticks=None):
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


def plot_3d(qx_array, qy_array, qz_array, image_array, index_list=None, crop=False):

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


def plot_3d_mm(x_array, y_array, z_array, image_array, index_list=None):

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

def plot_3d_grating(qx_array, qy_array, qz_array, image_array, qx, qy, qz, qx_0, qy_0, qz_0, index_list=None, crop=False):

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


def plot_sans(qy, qz, intensity):
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


def plot_penetration(qz_1d, depth_1d, legend=None,
                     x_max=None, y_min=None, y_max=None):
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


def plot_penetration_compare(qz1_1d, depth1_1d, qz2_1d, depth2_1d, legend=None,
                             x_max=None, y_min=None, y_max=None):

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


def plot_penetration_2d(ki_mesh, kf_mesh, depth_mesh, x_max=None, y_max=None):
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
