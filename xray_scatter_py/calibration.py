# -*- coding: utf-8 -*-
"""Calibration of the 2D images from x-ray scattering experiments

This module provides functions to calibrate the original 2D images in x-ray
scattering experiments. The calibration mainly involves calculating real-space
coordinates, scattering angles (theta), azimuth angles (phi), q-vectors, and
solid angles for each detector pixel. The module also provides functions to
normalize the intensity for each detector pixel, giving relative intensities
and absolute intensities.

The main functions in this module are used to:

get_mm: get Cartesian coordinates of each detector pixel in mm.
get_angle: get theta and azimuth angles for each detector pixel in radians.
get_q: get the q-vectors (qx, qy, qz) for each detector pixel in angstrom^-1.
get_chi: get the sample stage rotation around the chi axis in radians.
    The chi angle is the azimuthal angle with the largest integrated intensity
    for each measurement. This is used to correct the non-zero rotation of the
    sample stage around chi axis, the axis parallel to the incident beam. In a
    grazing incidence experiment, it is usually the out-of-plane direction
    that has the strongest total scattering intensity, as a result of the
    waveguiding effect in thin film samples. This feature is used to calculate
    the chi angle.
get_sr: get the solid angle (sr) for each detector pixel.
get_rel_intensity: Normalize the intensity for each detector pixel by exposure
    time and solid angle.
get_abs_intensity: Normalize the intensity for each detector pixel by exposure
    time, solid angle, sample transmission, sample thickness, and incidence
    beam intensity. The absolute intensity is also known as the differential
    scattering cross section at a certain q per sample thickness (cm) per
    solid angle (sr). This is a property only related to the sample, and is
    irrelavant to the facility used for the measurement.

These functions take in various parameters related to the X-ray scattering
geometry, such as the zero position of the detector (detx0), a list of
dictionaries containing the parameters of each measurement (params), and the
detector images (images) of multiple measurements.

These functions return arrays, containing information such as coordinates,
angles, q-vectors, chi angles, solid angles, and normalized intensities.
"""
import numpy as np

from xray_scatter_py import utils


def get_mm(detx0: float,
           params: list[dict],
           images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the Cartesian coordinates (x, y, z) of each detector pixel in mm.

    Args:
    - detx0 (float): Zero position of detector motor.
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'detx': Relative sample-detector distance in mm.
            - 'beamcenter_actual': Actual beam center position '[y z]' in mm.
            - 'pixelsize': Detector pixel size '[y z]' in mm.
    - images (np.ndarray): A 3D array of the original detector images.
        The first index is the serial number of measurement.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of three 3D arrays
        representing the x, y, z coordinate of each detector pixel in mm.
        Each array has the same shape as images.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])

    # initialize the arrays for the coordinates of each detector pixel
    x_array = np.ones_like(images, dtype=float)
    y_array = np.empty_like(images, dtype=float)
    z_array = np.empty_like(images, dtype=float)

    for i in range(images.shape[0]):

        image = images[i]
        param = params[i]

        # beam center coordinates in pixel
        beamcenter_y, beamcenter_z = map(
            float, param['beamcenter_actual'].strip('[]').split())
        # detector pixel size in mm
        pixelsize_y, pixelsize_z = map(
            float, param['pixelsize'].strip('[]').split())

        # coordinats of each detector pixel in mm
        mm_y = (np.arange(image.shape[0]) -
                beamcenter_y + 0.5) * pixelsize_y * (-1)
        mm_z = (np.arange(image.shape[1]) -
                beamcenter_z + 0.5) * pixelsize_z * (-1)
        y_array[i], z_array[i] = np.meshgrid(mm_y, mm_z)

        # For each measurement with the detector plane normal to the incidence
        # x-ray, the detector x coordinate is the same.
        x_array[i] = x_array[i] * (detx0 + float(param['detx']))

    return x_array, y_array, z_array


def get_angle(detx0: float,
              params: list[dict],
              images: np.ndarray,
              **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the theta and azimuth angles for each detector pixel in the images.

    Args:
    - detx0 (float): Zero position of detector motor.
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'detx': Relative sample-detector distance in mm.
            - 'beamcenter_actual': Actual beam center position '[y z]' in mm.
            - 'pixelsize': Detector pixel size '[y z]' in mm.
    - images (np.ndarray): A 3D array of the original detector images.
        The first index is the serial number of measurement.
    - kwargs:
        - chi (np.ndarray, optional): sample rotation around incidence x-ray.
            The 1D array has the same shape as the first dimension of images,
            representing the chi angle for each measurement. If not provided,
            default to an array of 0.5 * np.pi.

    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple of two 3D arrays representing the
        theta and azimuth angles of each detector pixel, with the same shape
        images.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])
    utils.validate_kwargs({'chi'}, kwargs)

    chi = kwargs.get('chi', np.ones(images.shape[0]) * 0.5 * np.pi)

    # initialize the arrays for the theta and azimuth angles
    theta_array = np.empty_like(images, dtype=float)
    azimuth_array = np.empty_like(images, dtype=float)

    for i in range(images.shape[0]):
        image = images[i]
        param = params[i]

        # beam center coordinates in pixel
        beamcenter_y, beamcenter_z = map(
            float, param['beamcenter_actual'].strip('[]').split())
        # detector pixel size in mm
        pixelsize_y, pixelsize_z = map(
            float, param['pixelsize'].strip('[]').split())
        # coordinats of each detector pixel in mm
        mm_y = (np.arange(image.shape[0])-beamcenter_y+0.5) * pixelsize_y
        mm_y = mm_y.reshape(-1, 1)
        mm_z = (np.arange(image.shape[1])-beamcenter_z+0.5) * pixelsize_z
        # get the theta and azimuthal angle of each detector pixel
        theta_array[i] = np.arctan(
            np.sqrt(mm_y**2 + mm_z**2) / (detx0+float(param['detx']))) / 2
        azimuth_array[i] = np.arctan2(mm_z, mm_y)
        # correct the azimuth angle for the sample rotation around chi axis
        azimuth_array[i] = azimuth_array[i] + 1.5 * np.pi - chi[i]

    return theta_array, azimuth_array


def get_q(detx0: float,
          params: list[dict],
          images: np.ndarray,
          **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the q-vectors (qx, qy, qz) for each detector pixel in the images.

    Args:
    - detx0 (float): Zero position of detector motor.
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'detx': Relative sample-detector distance in mm.
            - 'beamcenter_actual': Actual beam center position '[y z]' in mm.
            - 'pixelsize': Detector pixel size '[y z]' in mm.
            - 'wavelength': The wavelength of the x-ray beam in angstrom.
    - images (np.ndarray): A 3D array of the original detector images.
        The first index is the serial number of measurement.
    - kwargs:
        - chi (np.ndarray, optional): sample rotation around incidence x-ray.
            The 1D array has the same shape as the first dimension of images,
            representing the chi angle for each measurement. If not provided,
            default to an array of 0.5 * np.pi.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of three 3D arrays
        representing the qx, qy, and qz of each detector pixel, each with the
        same shape as images.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])
    utils.validate_kwargs({'chi'}, kwargs)

    chi = kwargs.get('chi', np.ones(images.shape[0]) * 0.5 * np.pi)

    theta_array, azimuth_array = get_angle(detx0, params, images, chi=chi)
    wavelength = float(params[0]['wavelength'])
    qx_array = 2 * np.pi / wavelength * (1 - np.cos(2 * theta_array))
    qy_array = 2 * np.pi / wavelength * \
        np.sin(2 * theta_array) * np.cos(azimuth_array)
    qz_array = 2 * np.pi / wavelength * \
        np.sin(2 * theta_array) * np.sin(azimuth_array)

    return qx_array, qy_array, qz_array


# -1 needs to be excluded
def get_chi(params: list[dict],
            azimuth_array: np.ndarray,
            images: np.ndarray,
            **kwargs) -> np.ndarray:
    """
    Get chi, the azimuthal angle with the largest integrated intensity.

    Args:
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'beamcenter_actual': Actual beam center position '[y z]' in mm.
    - azimuth_array (np.array): A 3D array of azimuth angle of each pixel.
    - images (np.ndarray): A 3D array of the original detector images.
        The first index is the serial number of measurement.
    - kwargs:
        - num_azimuth (int, optional): The number of azimuth values to screen.
        - center_mask (int, optional): Radius of the mask to exclude the
            spill-over incident beam.

    Returns:
        np.ndarray: 1D array containing the chi angle for each measurement.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])
    utils.validate_array_shape(images, azimuth_array)
    utils.validate_kwargs({'num_azimuth', 'center_mask'}, kwargs)

    num_azimuth = kwargs.get('num_azimuth', 720)
    center_mask = kwargs.get('center_mask', 20)

    # Azimuth angle array used to find the chi angle
    azimuth_1d = np.linspace(0, 2 * np.pi, num_azimuth)
    chi_1d = np.empty(images.shape[0])

    for i in range(images.shape[0]):

        print(i, 'of', images.shape[0])

        image = images[i]
        param = params[i]
        azimuth = azimuth_array[i]

        # beam center coordinates in pixel
        beamcenter_y, beamcenter_z = map(
            float, param['beamcenter_actual'].strip('[]').split())
        # dectector pixel coordinates corrected by the beam center
        diff_center_y = (np.arange(image.shape[0]) - beamcenter_y)**2
        diff_center_z = (np.arange(image.shape[1]) - beamcenter_z)**2
        # mask to exclude the spill-over incident beam
        bool_center = ((diff_center_y[:, np.newaxis] +
                        diff_center_z[np.newaxis, :]) > center_mask**2)

        # mask for selecting the pixels at a certain azimuth angle
        azimuth_diff = azimuth[:, :, np.newaxis] - \
            azimuth_1d[np.newaxis, np.newaxis, :]
        bool_azimuth = np.logical_and(azimuth_diff > -np.pi / num_azimuth,
                                      azimuth_diff < np.pi / num_azimuth)

        # final mask for selecting the pixels for integration
        bool_image = np.logical_and(bool_center[:, :, np.newaxis],
                                    bool_azimuth)

        # get the integrated intensity for every azimuthal angle
        sum_azimuth = np.sum(image[:, :, np.newaxis] * bool_image, axis=(0, 1))

        # assign azimuthal angle with maximum integrated intensity as chi
        chi_1d[i] = azimuth_1d[np.argmax(sum_azimuth)]

    return chi_1d


def get_sr(detx0: float,
           params: list[dict],
           theta_array: np.ndarray) -> np.ndarray:
    """
    Get the solid angle (sr) for each pixel of the detector images.

    Args:
    - detx0 (float): Zero position of detector motor.
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'detx': Relative sample-detector distance in mm.
            - 'pixelsize': Detector pixel size '[y z]' in mm.
    - theta_array (np.ndarray): A 3D array of scattering angle of each pixel.
        The first index is the serial number of measurement.

    Returns:
    - np.ndarray: A 3D array containing the solid angles for each pixel.
        The array has the same shape as the input theta_array.
    """

    utils.validate_array_dimension(theta_array, 3)
    utils.validate_list_len(params, theta_array.shape[0])

    sr_array = np.empty_like(theta_array)

    for i in range(theta_array.shape[0]):
        # detector pixel size in mm
        pixelsize_y, pixelsize_z = map(
            float, params[i]['pixelsize'].strip('[]').split())
        # relative sample-detector distance in mm
        detx = float(params[i]['detx'])
        sr_array[i] = pixelsize_y * pixelsize_z / \
            (detx0 + detx)**2 * (np.cos(2 * theta_array[i]))**3

    return sr_array


def get_rel_intensity(params: list[dict],
                      images: np.ndarray,
                      sr_array: np.ndarray) -> np.ndarray:
    """
    Normalize the intensity by exposure time and solid angle.

    Args:
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'det_exposure_time': The exposure time in seconds.
    - images (np.ndarray): A 3D array of the original detector images.
        The first index is the serial number of measurement.
    - sr_array (np.ndarray): A 3D array of the solid angles for each pixel.

    Returns:
    - np.ndarray: A 3D array containing the normalized relative intensities.
        The array has the same shape as the input images.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])
    utils.validate_array_shape(images, sr_array)

    time = np.empty(images.shape[0])
    for i, param in enumerate(params):
        time[i] = float(param['det_exposure_time'])

    # invalid pixels all have intensity of -1
    images_bool = images == -1
    images = images / time[:, np.newaxis, np.newaxis] / sr_array
    images[images_bool] = -1

    return images


def get_abs_intensity(params: list[dict],
                      images: np.ndarray) -> np.ndarray:
    """
    Normalize the intensity by transmission factor, sample thickness, and
    incident intensity, assuming the intensity is already normalized by
    exposure time and solid angle.

    Args:
    - params (list[dict]): Each dict contains parameters of each measurement.
        Each dictionary should contain the following keys with string values:
            - 'sample_transfact': The transmission factor of the sample.
            - 'sample_thickness': The thickness of the sample in mm.
            - 'saxsconf_Izero': The incident intensity.
    - images (np.ndarray): A 3D array of the original detector images.
        The first index is the serial number of measurement.

    Returns:
    - np.ndarray: A 3D array containing the absolute intensities.
        The array has the same shape as the input images.
    """

    utils.validate_array_dimension(images, 3)
    utils.validate_list_len(params, images.shape[0])

    transmission = np.empty(images.shape[0])
    thickness = np.empty(images.shape[0])
    incidence = np.empty(images.shape[0])
    for i, param in enumerate(params):
        transmission[i] = float(param['sample_transfact'])
        thickness[i] = float(param['sample_thickness'])
        incidence[i] = float(param['saxsconf_Izero'])

    # invalid pixels all have intensity of -1
    images_bool = images == -1
    images = images / transmission[:, np.newaxis, np.newaxis] /\
        thickness[:, np.newaxis, np.newaxis] / \
        incidence[:, np.newaxis, np.newaxis]
    images[images_bool] = -1

    return images
