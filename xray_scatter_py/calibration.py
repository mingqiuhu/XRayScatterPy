# xray_scatter_py/calibration.py

import numpy as np

def calculate_mm(DETX0, params_dict_list, image_array):
    x_array = np.ones_like(image_array)
    y_array = np.empty_like(image_array)
    z_array = np.empty_like(image_array)
    for i in range(image_array.shape[0]):
        params_dict = params_dict_list[i]
        image = image_array[i]
        detx = float(params_dict['detx'])
        beamcenter_x, beamcenter_y = map(float, params_dict['beamcenter_actual'].strip('[]').split())
        pixelsize_x, pixelsize_y = map(float, params_dict['pixelsize'].strip('[]').split())
        mm_x = (np.arange(image.shape[0]) - beamcenter_x + 0.5) * pixelsize_x * (-1)
        mm_y = (np.arange(image.shape[1]) - beamcenter_y + 0.5) * pixelsize_y * (-1)
        z_array[i], y_array[i] = np.meshgrid(mm_y, mm_x)
        x_array[i] = x_array[i] * (DETX0 + detx)
    return x_array, y_array, z_array


def calculate_angle(detx0, params_dict_list, image_array, rho=None):
    """
    Calculate the theta and azimuth angles for each image in the image_array.
    
    Args:
        detx0 (float): Detector distance.
        params_dict_list (list): List of dictionaries containing the parameters for each image.
        image_array (np.ndarray): Array containing the images.
    
    Returns:
        tuple: Two numpy arrays, theta_array and azimuth_array, containing the theta and azimuth angles, respectively.
    """
    theta_array = np.empty_like(image_array, dtype=float)
    azimuth_array = np.empty_like(image_array, dtype=float)

    if rho is None:
        rho = np.ones(image_array.shape[0]) * 0.5 * np.pi

    for i in range(image_array.shape[0]):
        params_dict = params_dict_list[i]
        image = image_array[i]

        detx = float(params_dict['detx'])
        beamcenter_x, beamcenter_y = map(float, params_dict['beamcenter_actual'].strip('[]').split())
        pixelsize_x, pixelsize_y = map(float, params_dict['pixelsize'].strip('[]').split())
        mm_x = ((np.arange(image.shape[0]) - beamcenter_x + 0.5) * pixelsize_x).reshape(-1, 1)
        mm_y = (np.arange(image.shape[1]) - beamcenter_y + 0.5) * pixelsize_y

        theta_array[i] = np.arctan(np.sqrt(mm_x**2 + mm_y**2) / (detx0 + detx)) / 2
        azimuth_array[i] = np.arctan2(mm_y, mm_x)
        azimuth_array[i] = azimuth_array[i] + 1.5 * np.pi - rho[i]
    return theta_array, azimuth_array

def calculate_q(detx0, params_dict_list, image_array, rho=None):
    """
    Calculate the Q-vectors (qx, qy, qz) for given images using the detector distance and parameters.
    
    Args:
        detx0 (float): Detector distance.
        params_dict_list (list): List of dictionaries containing the parameters for each image.
        image_array (np.ndarray): Array containing the images.
    
    Returns:
        tuple: Three numpy arrays, qx_array, qy_array, and qz_array, containing the Q-vector components, respectively.
    """
    theta_array, azimuth_array = calculate_angle(detx0, params_dict_list, image_array, rho)
    
    wavelength = float(params_dict_list[0]['wavelength'])
    qx_array = 2 * np.pi / wavelength * (1 - np.cos(2 * theta_array))
    qy_array = 2 * np.pi / wavelength * np.sin(2 * theta_array) * np.cos(azimuth_array)
    qz_array = 2 * np.pi / wavelength * np.sin(2 * theta_array) * np.sin(azimuth_array)
    
    return qx_array, qy_array, qz_array

# -1 needs to be excluded
def calculate_rho(params_dict_list, azimuth_array, image_array, num_azimuth=720, center_mask=20):
    """
    Calculate rho values from a list of parameter dictionaries, azimuth array, and image array.

    Args:
        params_dict_list (list): List of dictionaries containing parameters.
        azimuth_array (numpy.array): 2D array containing azimuth values.
        image_array (numpy.array): 3D array containing image data.
        num_azimuth (int, optional): Number of azimuth values. Defaults to 90.

    Returns:
        numpy.array: 1D array of rho values.
    """
    list_azimuth = np.linspace(0, 2 * np.pi, num_azimuth)
    list_rho = np.empty(image_array.shape[0])

    for i in range(image_array.shape[0]):
        print(i, 'of', image_array.shape[0])
        params_dict = params_dict_list[i]
        azimuth = azimuth_array[i]
        image = image_array[i]

        beamcenter_x, beamcenter_y = map(float, params_dict['beamcenter_actual'].strip('[]').split())
        bool_center_x = (np.arange(image.shape[0]) - beamcenter_x)**2
        bool_center_y = (np.arange(image.shape[1]) - beamcenter_y)**2
        bool_center = ((bool_center_x[:, np.newaxis] + bool_center_y[np.newaxis, :]) > center_mask**2)

        azimuth_diff = azimuth[:, :, np.newaxis] - list_azimuth[np.newaxis, np.newaxis, :]
        bool_azimuth = np.logical_and(azimuth_diff > -np.pi / num_azimuth, azimuth_diff < np.pi / num_azimuth)

        bool_image = np.logical_and(bool_center[:, :, np.newaxis], bool_azimuth)

        sum_azimuth = np.sum(image[:, :, np.newaxis] * bool_image, axis=(0, 1))

        list_rho[i] = list_azimuth[np.argmax(sum_azimuth)]

    return list_rho

def calculate_omega(detx0, params_dict_list, theta_array):
    omega = np.empty_like(theta_array)
    for i in range(theta_array.shape[0]):
        pixelsize_x, pixelsize_y = map(float, params_dict_list[i]['pixelsize'].strip('[]').split())
        detx = float(params_dict_list[i]['detx'])
        omega[i] = pixelsize_x * pixelsize_y / (detx0 + detx)**2 * (np.cos(2 * theta_array[i]))**3
    return omega

def calibrate_rel_intensity(params_dict_list, image_array, omega):
    time = np.empty(image_array.shape[0])
    for i in range(image_array.shape[0]):
        time[i] = float(params_dict_list[i]['det_exposure_time'])
    image_array_bool = image_array==-1
    image_array = image_array / time[:, np.newaxis, np.newaxis] / omega
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