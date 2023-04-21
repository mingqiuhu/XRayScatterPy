# xray_scatter_py/calibration.py

import numpy as np

def calculate_mm(detx0: float, params_dict_list: list[dict],
                 image_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the coordinate of each detector pixel in the unit of mm, based on the zero position
    of detector, the paramters of each measurement, and the shape of the original detector images.

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

    Returns:
    - tuple: A tuple of three 3D arrays representing the coordinate of each detector pixel in mm.
        Each array has the same shape as the input image_array.
    """

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
                    kai: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
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
    - kai (np.ndarray, optional): Array of initial azimuth angles. 
        If None, an array of 0.5 * pi is used.

    Returns:
    - tuple: A tuple of two 3D arrays representing the theta and azimuth angles of each detector.
        Each array has the same shape as the input image_array.
    """
    # initialize the output arrays containing the theta and azimuth angles of each detector pixel
    theta_array = np.empty_like(image_array, dtype=float)
    azimuth_array = np.empty_like(image_array, dtype=float)

    # initialize the kai angle array if not provided.
    if kai is None:
        kai = np.ones(image_array.shape[0]) * 0.5 * np.pi

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
                kai: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    - kai (np.ndarray, optional): Array of initial azimuth angles. 
        If None, an array of 0.5 * pi is used.

    Returns:
    - tuple: A tuple of three 3D arrays representing the qx, qy, qz in A-1 of each detector pixel.
        Each array has the same shape as the input image_array.
    """

    # Calculate the theta and azimuthal angle of each detector pixel
    theta_array, azimuth_array = calculate_angle(detx0, params_dict_list, image_array, kai)

    # Calculate the q-vectors for each detector pixel, assuming the wavelength is the same for all.
    wavelength = float(params_dict_list[0]['wavelength'])
    qx_array = 2 * np.pi / wavelength * (1 - np.cos(2 * theta_array))
    qy_array = 2 * np.pi / wavelength * np.sin(2 * theta_array) * np.cos(azimuth_array)
    qz_array = 2 * np.pi / wavelength * np.sin(2 * theta_array) * np.sin(azimuth_array)

    return qx_array, qy_array, qz_array

# -1 needs to be excluded
def calculate_kai(params_dict_list: list[dict], azimuth_array: np.ndarray, image_array: np.ndarray,
                  num_azimuth: int = 720, center_mask: int = 20):
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
    - num_azimuth (int, optional): The number of azimuth values to consider.
        Defaults to 720
    - center_mask (int, optional): Radius of the mask used to exclude spill-over incident beam.
        Defaults to 20

    Returns:
        np.ndarray: A 1D array containing the computed kai angles for each measurement.
    """

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


def calculate_sr(detx0: float, params_dict_list: list[dict], theta_array: np.ndarray):
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

    # initialize the output solid angle array wih the same shape as theta_array
    sr_array = np.empty_like(theta_array)

    # iterate through each measurement
    for i in range(theta_array.shape[0]):
        # get the detector pixel size and the sample-detector distance
        pixelsize_y, pixelsize_z = map(float,
                                       params_dict_list[i]['pixelsize'].strip('[]').split())
        detx = float(params_dict_list[i]['detx'])
        # calculate the solid angle for each pixel
        sr_array[i] = pixelsize_y * pixelsize_z / (detx0 + detx)**2 * (np.cos(2 * theta_array[i]))**3

    return sr_array

def calibrate_rel_intensity(params_dict_list: list[dict], image_array: np.ndarray,
                            sr_array: np.ndarray):
    # get the exposure time of each measurement
    time = np.empty(image_array.shape[0])
    for i in range(image_array.shape[0]):
        time[i] = float(params_dict_list[i]['det_exposure_time'])
    image_array_bool = image_array==-1
    image_array = image_array / time[:, np.newaxis, np.newaxis] / sr_array
    image_array[image_array_bool] = -1
    return image_array

def calibrate_abs_intensity(params_dict_list, image_array):
    transmission = np.empty(image_array.shape[0])
    thickness = np.empty(image_array.shape[0])
    incidence = np.empty(image_array.shape[0])
    for i in range(image_array.shape[0]):
        transmission[i] = float(params_dict_list[i]['sample_transfact'])
        thickness[i] = float(params_dict_list[i]['sample_thickness'])
        incidence[i] = float(params_dict_list[i]['saxsconf_Izero'])
    image_array_bool = image_array==-1
    image_array = image_array / transmission[:, np.newaxis, np.newaxis] / thickness[:, np.newaxis, np.newaxis] / incidence[:, np.newaxis, np.newaxis]
    image_array[image_array_bool] = -1
    return image_array