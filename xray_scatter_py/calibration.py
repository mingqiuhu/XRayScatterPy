"""
This module provides functions to calibrate the 2D detector of an x-ray scatttering experiment. 
The calibration mainly involves calculating real-spece coordinates (mm), scattering angles,
azimuth angles, q-vectors, and solid angles for each detector pixel. The module also provides
functions to normalize the intensity for each detector pixel, giving relative intensities and
absolute intensities.

The main functions in this module are used to:

1. calculate_mm: Calculate the real space coordinates of each detector pixel in the unit of mm.
2. calculate_angle: Calculate the theta and azimuth angles for each detector pixel.
3. calculate_q: Calculate the q-vectors (qx, qy, qz) for each detector pixel.
4. calculate_kai: Calculate the azimuthal angle with the largest integrated intensity, kai,
   for each image. This is used to correct the non-zero rotation of the sample stage around kai
   axis, the axis paralell to the incident beam. In a grazing incicence experiment, it is usually
   the out-of-plane direction that has the strongest total scattering intensity, as a result of
   the waveguiding effect in thin film samlpes.
5. calculate_sr: Calculate the solid angle (sr) for each detector pixel. The solid angle is defined
   as the area on the surface of a sphere normalized by square of radius of the sphere.
6. calculate_rel_intensity: Normalize the intensity for each detector pixel by exposure time and
   solid angle.
7. calculate_abs_intensity: Normalize the intensity for each detector pixel by exposure time, solid
   angle, sample transmission, sample thickness, and incidence beam intensity. The absolute
   intensity is also known as the differential scattering cross section at a certain q per
   sample thickness (cm) per solid angle (sr).

These functions take in various parameters related to the X-ray scattering geometry, such as
the zero position of the detector (detx0), a list of dictionaries containing the parameters of
each measurement (params_dict_list), and the 3D array of detector images (image_array).

These functions return arrays, containing information such as coordinates, angles, q-vectors,
kai angles, solid angles, and normalized intensities.
"""


import numpy as np
from xray_scatter_py import utils


def calculate_mm(detx0: float, params_dict_list: list[dict],
                 image_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the coordinate of each detector pixel in the unit of mm, based on the zero position
    of detector, the paramters of each measurement, and the shape of the original detector images.

    Args:
    - detx0 (float): Zero position of detector motor. detx0+detx is the sample-detector distance.
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following keys:
            - 'detx': The relative sample-detector distance as a string in the unit of mm.
            - 'beamcenter_actual': The actual beam center position as a string '[y z]'.
            - 'pixelsize': The pixel size as a string '[y z]' in the unit of mm.
    - image_array (np.ndarray): A 3D array of the input images. 
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.

    Returns:
    - tuple: A tuple of three 3D arrays representing the coordinate of each detector pixel in mm.
        Each array has the same shape as the input image_array.
    """

    utils.validate_array_dimension(image_array, 3)
    utils.validate_list_len(params_dict_list, image_array.shape[0])

    # initialize the output arrays containing the x, y, z coordinates of each detector pixel
    x_array = np.ones_like(image_array, dtype=float)
    y_array = np.empty_like(image_array, dtype=float)
    z_array = np.empty_like(image_array, dtype=float)

    # iterate through each measurement
    for i in range(image_array.shape[0]):

        # original detector image for the current measurement
        image = image_array[i]
        # parameters for the current measurement
        params_dict = params_dict_list[i]

        # beam center in the unit of pixel
        beamcenter_y, beamcenter_z = map(float,
                                         params_dict['beamcenter_actual'].strip('[]').split())
        # detector pixel size in the unit of mm
        pixelsize_y, pixelsize_z = map(float, params_dict['pixelsize'].strip('[]').split())

        # Calculate the coordinats of each detector pixel in the unit of mm.
        mm_y = (np.arange(image.shape[0]) - beamcenter_y + 0.5) * pixelsize_y * (-1)
        mm_z = (np.arange(image.shape[1]) - beamcenter_z + 0.5) * pixelsize_z * (-1)
        y_array[i], z_array[i] = np.meshgrid(mm_y, mm_z)

        # For each measurement, the detector x coordinate is the same.
        # detx is the relative sample-detector distance.
        x_array[i] = x_array[i] * (detx0 + float(params_dict['detx']))

    return x_array, y_array, z_array


def calculate_angle(detx0: float, params_dict_list: list[dict], image_array: np.ndarray,
                    **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the theta and azimuth angles for each detector pixel in the image_array.

    Args:
    - detx0 (float): Zero position of detector. detx0+detx is the sample-detector distance.
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following keys:
            - 'detx': The relative sample-detector distance as a string.
            - 'beamcenter_actual': The actual beam center position as a string '[y z]'.
            - 'pixelsize': The pixel size as a string '[y z]'.
    - image_array (np.ndarray): A 3D array of the input images. 
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.
    - kwargs:
        - kai (np.ndarray, optional): sample stage rotation around the kai axis paralell to x-ray.
            The 1D array has the same shape as the first dimension of the 3D image array,
            representing the kai angle for each measurement. If not provided, default to an array
            of 0.5 * np.pi.

    Returns:
    - tuple: A tuple of two 3D arrays representing the theta and azimuth angles of each detector.
        Each array has the same shape as the input image_array.
    """

    utils.validate_array_dimension(image_array, 3)
    utils.validate_list_len(params_dict_list, image_array.shape[0])
    utils.validate_kwargs({'kai'}, kwargs)

    # initialize the kai angle array if not provided.
    kai = kwargs.get('kai', np.ones(image_array.shape[0]) * 0.5 * np.pi)

    # initialize the output arrays containing the theta and azimuth angles of each detector pixel
    theta_array = np.empty_like(image_array, dtype=float)
    azimuth_array = np.empty_like(image_array, dtype=float)

    # iterate through each measurement
    for i in range(image_array.shape[0]):
        # original detector image for the current measurement
        image = image_array[i]
        # parameters for the current measurement
        params_dict = params_dict_list[i]

        # beam center in the unit of pixel
        beamcenter_y, beamcenter_z = map(float,
                                         params_dict['beamcenter_actual'].strip('[]').split())
        # detector pixel size in the unit of mm
        pixelsize_y, pixelsize_z = map(float, params_dict['pixelsize'].strip('[]').split())
        # Calculate the coordinats of each detector pixel in the unit of mm
        mm_y = ((np.arange(image.shape[0]) - beamcenter_y + 0.5) * pixelsize_y).reshape(-1, 1)
        mm_z = (np.arange(image.shape[1]) - beamcenter_z + 0.5) * pixelsize_z
        # Calculate the theta and azimuthal angle of each detector pixel
        theta_array[i] = np.arctan(np.sqrt(mm_y**2 + mm_z**2) / (detx0 +
                                                                 float(params_dict['detx']))) / 2
        azimuth_array[i] = np.arctan2(mm_z, mm_y)
        azimuth_array[i] = azimuth_array[i] + 1.5 * np.pi - kai[i]

    return theta_array, azimuth_array


def calculate_q(detx0: float, params_dict_list: list[dict], image_array: np.ndarray,
                **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the q-vectors (qx, qy, qz) for each pixel of the detector image.

    Args:
    - detx0 (float): Zero position of detector. detx0+detx is the sample-detector distance.
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following keys:
            - 'detx': The relative sample-detector distance as a string.
            - 'beamcenter_actual': The actual beam center position as a string '[y z]'.
            - 'pixelsize': The pixel size as a string '[y z]'.
            - 'wavelength': The wavelength of the x-ray beam as a string.
    - image_array (np.ndarray): A 3D array of the input images. 
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.
    - kwargs:
        - kai (np.ndarray, optional): sample stage rotation around the kai axis paralell to x-ray.
            The 1D array has the same shape as the first dimension of the 3D image array.
            If not provided, default to an array of 0.5 * np.pi.

    Returns:
    - tuple: A tuple of three 3D arrays representing the qx, qy, qz in A-1 of each detector pixel.
        Each array has the same shape as the input image_array.
    """

    utils.validate_array_dimension(image_array, 3)
    utils.validate_list_len(params_dict_list, image_array.shape[0])
    utils.validate_kwargs({'kai'}, kwargs)

    # initialize the kai angle array if not provided.
    kai = kwargs.get('kai', np.ones(image_array.shape[0]) * 0.5 * np.pi)

    # Calculate the theta and azimuthal angle of each detector pixel
    theta_array, azimuth_array = calculate_angle(detx0, params_dict_list, image_array, kai=kai)

    # Calculate the q-vectors for each detector pixel, assuming the wavelength is the same for all.
    wavelength = float(params_dict_list[0]['wavelength'])
    qx_array = 2 * np.pi / wavelength * (1 - np.cos(2 * theta_array))
    qy_array = 2 * np.pi / wavelength * np.sin(2 * theta_array) * np.cos(azimuth_array)
    qz_array = 2 * np.pi / wavelength * np.sin(2 * theta_array) * np.sin(azimuth_array)

    return qx_array, qy_array, qz_array


# -1 needs to be excluded
def calculate_kai(params_dict_list: list[dict], azimuth_array: np.ndarray, image_array: np.ndarray,
                  **kwargs) -> np.ndarray:
    """
    Calculate kai, the azimuthal angle with the largest integrated intensity, for each image.

    Args:
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following keys:
            - 'beamcenter_actual': The actual beam center position as a string '[y z]'.
    - azimuth_array (np.array): A 3D array of the azimuth angle of each pixel of the detector image. 
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.
    - image_array (np.ndarray): A 3D array of the input images.
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.
    - kwargs:
        - num_azimuth (int, optional): The number of azimuth values to screen.
            If not provided, default to 720.
        - center_mask (int, optional): Radius of the mask used to exclude spill-over incident beam.
            If not provided, default to 20 (pixels)

    Returns:
        np.ndarray: A 1D array containing the computed kai angles for each measurement.
    """

    utils.validate_array_dimension(image_array, 3)
    utils.validate_list_len(params_dict_list, image_array.shape[0])
    utils.validate_array_shape(image_array, azimuth_array)
    utils.validate_kwargs({'num_azimuth', 'center_mask'}, kwargs)

    num_azimuth = kwargs.get('num_azimuth', 720)
    center_mask = kwargs.get('center_mask', 20)

    # initialize the 1D array of azimuth angles from 0 to 2pi for calculating the kai values
    azimuth_1d = np.linspace(0, 2 * np.pi, num_azimuth)
    # initialize the output 1D array of kai values of each measurement
    kai_1d = np.empty(image_array.shape[0])

    # iterate through each measurement
    for i in range(image_array.shape[0]):

        # print the progress
        print(i, 'of', image_array.shape[0])

        # detector image, measurement parameters, and azimuth angle array for current measurement
        image = image_array[i]
        params_dict = params_dict_list[i]
        azimuth = azimuth_array[i]

        # calculate the boolean array to exclude the spill-over incident beam from calculating kai
        beamcenter_y, beamcenter_z = map(float,
                                         params_dict['beamcenter_actual'].strip('[]').split())
        diff_center_y = (np.arange(image.shape[0]) - beamcenter_y)**2
        diff_center_z = (np.arange(image.shape[1]) - beamcenter_z)**2
        bool_center = ((diff_center_y[:, np.newaxis] + diff_center_z[np.newaxis, :]) >
                       center_mask**2)

        # calculate the boolean mask for the pixels in a certain azimuthal angle range
        azimuth_diff = azimuth[:, :, np.newaxis] - azimuth_1d[np.newaxis, np.newaxis, :]
        bool_azimuth = np.logical_and(azimuth_diff > -np.pi / num_azimuth,
                                      azimuth_diff < np.pi / num_azimuth)

        # calculate the final boolean mask for the integrated intensity of every azimuthal angle
        bool_image = np.logical_and(bool_center[:, :, np.newaxis], bool_azimuth)

        # calculate the integrated intensity in every azimuthal angle
        sum_azimuth = np.sum(image[:, :, np.newaxis] * bool_image, axis=(0, 1))

        # find the azimuthal angle with the maximum integrated intensity and assign as kai angle
        kai_1d[i] = azimuth_1d[np.argmax(sum_azimuth)]

    return kai_1d


def calculate_sr(detx0: float, params_dict_list: list[dict], theta_array: np.ndarray) -> np.ndarray:
    """
    Calculate the solid angle (sr) for each pixel of the detector images.

    Args:
    - detx0 (float): Zero position of detector. detx0+detx is the sample-detector distance.
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following keys:
            - 'detx': The relative sample-detector distance in mm as a string.
            - 'pixelsize': The pixel size in mm as a string '[y z]'.
    - theta_array (np.ndarray): A 3D array of the scattering angle of each pixel of detector image.
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.

    Returns:
    - np.ndarray: A 3D array containing the computed solid angles for each pixel of detector image.
        The array has the same shape as the input theta_array.
    """

    utils.validate_array_dimension(theta_array, 3)
    utils.validate_list_len(params_dict_list, theta_array.shape[0])

    # initialize the output solid angle array wih the same shape as theta_array
    sr_array = np.empty_like(theta_array)

    # iterate through each measurement
    for i in range(theta_array.shape[0]):
        # get the detector pixel size and the sample-detector distance
        pixelsize_y, pixelsize_z = map(float,
                                       params_dict_list[i]['pixelsize'].strip('[]').split())
        detx = float(params_dict_list[i]['detx'])
        # calculate the solid angle for each pixel
        sr_array[i] = pixelsize_y * pixelsize_z / (detx0 + detx)**2 * (np.cos(2 *
                                                                              theta_array[i]))**3

    return sr_array


def calculate_rel_intensity(params_dict_list: list[dict], image_array: np.ndarray,
                            sr_array: np.ndarray) -> np.ndarray:
    """
    Normalize the intensity for each pixel of the detector image by exposure time and solid angle.

    Args:
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following key:
            - 'det_exposure_time': The exposure time of the detector as a string.
    - image_array (np.ndarray): A 3D array of the input images.
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.
    - sr_array (np.ndarray): A 3D array of the solid angles for each pixel of the detector image.
        The array has the same shape as the input image_array.

    Returns:
    - np.ndarray: A 3D array containing the normalized relative intensities for each pixel of the
        detector image. The array has the same shape as the input image_array.
    """

    utils.validate_array_dimension(image_array, 3)
    utils.validate_list_len(params_dict_list, image_array.shape[0])
    utils.validate_array_shape(image_array, sr_array)

    # get the exposure time of each measurement
    time = np.empty(image_array.shape[0])
    for i in range(image_array.shape[0]):
        time[i] = float(params_dict_list[i]['det_exposure_time'])

    # normalize the intensity by exposure time and solid angle, keeping the invalid pixels as -1
    image_array_bool = image_array==-1
    image_array = image_array / time[:, np.newaxis, np.newaxis] / sr_array
    image_array[image_array_bool] = -1

    return image_array


def calculate_abs_intensity(params_dict_list: list[dict], image_array: np.ndarray) -> np.ndarray:
    """
    Calculate absolute intensity for each pixel of the detector image by normalizing the intensity
    by transmission factor, sample thickness, and incident intensity.

    Args:
    - params_dict_list (list[dict]): List of dictionaries containing parameters of each measurement.
        Each dictionary should contain the following keys:
            - 'sample_transfact': The transmission factor of the sample as a string.
            - 'sample_thickness': The thickness of the sample as a string.
            - 'saxsconf_Izero': The incident intensity as a string.
    - image_array (np.ndarray): A 3D array of the input images.
        The first index is the serial number of measurement. The second and third indices are the
        y and z indices of the detector image.

    Returns:
    - np.ndarray: A 3D array containing the absolute intensities for each pixel of the detector
        image. The array has the same shape as the input image_array.
    """

    utils.validate_array_dimension(image_array, 3)
    utils.validate_list_len(params_dict_list, image_array.shape[0])

    # get the transmission factor, sample thickness, and incident intensity of each measurement
    transmission = np.empty(image_array.shape[0])
    thickness = np.empty(image_array.shape[0])
    incidence = np.empty(image_array.shape[0])
    for i in range(image_array.shape[0]):
        transmission[i] = float(params_dict_list[i]['sample_transfact'])
        thickness[i] = float(params_dict_list[i]['sample_thickness'])
        incidence[i] = float(params_dict_list[i]['saxsconf_Izero'])

    # Normalize the intensity by transmission factor, sample thickness, and incident intensity,
    image_array_bool = image_array==-1
    image_array = image_array / transmission[:, np.newaxis, np.newaxis] /\
        thickness[:, np.newaxis, np.newaxis] / incidence[:, np.newaxis, np.newaxis]
    image_array[image_array_bool] = -1

    return image_array
