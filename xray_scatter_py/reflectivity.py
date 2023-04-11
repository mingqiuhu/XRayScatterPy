# xray_scatter_py/reflectivity.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad

def calculate_relative_reflectivity(qy_fwhm, interval_degree, qy_array, qz_array, params_dict_list, image_array):
    interval_degree = np.radians(interval_degree)
    qz_1d = np.empty(image_array.shape[0])
    reflectivity_array = np.empty(image_array.shape[0])
    spillover_array = np.empty(image_array.shape[0])
    total_array = np.empty(image_array.shape[0])
    for i in range(image_array.shape[0]):
        wavelength = float(params_dict_list[0]['wavelength'])
        incidence_degree = np.radians(float(params_dict_list[i]['sample_angle1']))
        time = float(params_dict_list[i]['det_exposure_time'])
        
        qz_upper = 4*np.pi*np.sin(incidence_degree + 0.25*interval_degree)/wavelength
        qz_lower = 4*np.pi*np.sin(incidence_degree - 0.25*interval_degree)/wavelength
        qz_bool = np.logical_and((qz_array[i]**2 + qy_array[i]**2) >= qz_lower**2, 
                                (qz_array[i]**2 + qy_array[i]**2) <= qz_upper**2)
        qy_bool = (np.abs(qy_array[i]) <= qy_fwhm)
        image_bool = (image_array[i]!=-1)
        q_center_bool = ((qz_array[i]**2 + qy_array[i]**2) <= 0.0052**2)
        
        qz_1d[i] = 4*np.pi*np.sin(incidence_degree) / wavelength
        reflectivity_array[i] = np.sum(image_array[i] * qz_bool * qy_bool * image_bool) / time
        spillover_array[i] = np.sum(image_array[i] * image_bool * q_center_bool) / time
        total_array[i] = np.sum(image_array[i] * image_bool) / time
        
    return qz_1d, reflectivity_array, spillover_array, total_array

def calculate_normalized_reflectivity(params_dict_list, qz_1d, reflectivity_array, spillover_array, fixed_sigma = 0.064):
    wavelength = float(params_dict_list[0]['wavelength'])
    theta_1d = np.arcsin(qz_1d * wavelength / 4 / np.pi)

    # Define the integrand function
    def integrand(y_sample, incidence_intensity):
        return incidence_intensity / (fixed_sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * y_sample ** 2 / fixed_sigma ** 2)

    # Define the function to fit
    def spillover_func(theta, incidence_intensity, len_sample):
        fitted_spillover = []
        for t in theta:
            integral, _ = quad(integrand, 0, len_sample * np.sin(t) / 2, args=(incidence_intensity,))
            fitted_spillover.append(0.5 * incidence_intensity - integral)
        return np.array(fitted_spillover)

    # Fit the function to the data
    initial_guess = [2.5e6, 10]  # Initial guess for incidence_intensity, len_sample
    optimal_params, _ = curve_fit(spillover_func, theta_1d[:-5], spillover_array[:-5], p0=initial_guess)
    incidence_intensity, len_sample = optimal_params
    print("Optimal I0:", incidence_intensity)
    print("Optimal L0:", len_sample)

    fitted_spillover = []
    beam_on_sample = []
    for t in theta_1d:
        integral, _ = quad(integrand, 0, len_sample * np.sin(t) / 2, args=(incidence_intensity,))
        fitted_spillover.append(0.5 * incidence_intensity - integral)
        beam_on_sample.append(2 * integral)
    normalized_reflectivity = reflectivity_array / np.array(beam_on_sample)
    return normalized_reflectivity, np.array(fitted_spillover)


def plot_specular_reflectivity(incidence_angles, reflectivity):
    """
    Plot the specular reflectivity as a function of the incidence angle.

    Args:
        incidence_angles (numpy.ndarray): Incidence angles.
        reflectivity (numpy.ndarray): Specular reflectivity values.
    """
    plt.figure()
    plt.plot(incidence_angles, reflectivity)
    plt.xlabel('Incidence Angle')
    plt.ylabel('Specular Reflectivity')
    plt.title('Specular Reflectivity vs Incidence Angle')
    plt.show()

def extract_off_specular_scattering(images, incidence_angles, exit_angle):
    """
    Extract off-specular scattering from the calibrated 2D GISAXS images.

    Args:
        images (list): A list of calibrated 2D GISAXS images.
        incidence_angles (list): A list of corresponding incidence angles.
        exit_angle (float): The fixed non-zero angle between the incidence and exit beam.

    Returns:
        numpy.ndarray: Off-specular scattering intensity extracted from the images.
    """
    # Replace this with the actual method to extract off-specular scattering from the images
    off_specular_scattering = np.array([np.sum(image) for image in images])

    return off_specular_scattering

def plot_off_specular_scattering(incidence_angles, scattering_intensity):
    """
    Plot the off-specular scattering intensity as a function of the incidence angle.

    Args:
        incidence_angles (numpy.ndarray): Incidence angles.
        scattering_intensity (numpy.ndarray): Off-specular scattering intensity values.
    """
    plt.figure()
    plt.plot(incidence_angles, scattering_intensity)
    plt.xlabel('Incidence Angle')
    plt.ylabel('Off-specular Scattering Intensity')
    plt.title('Off-specular Scattering Intensity vs Incidence Angle')
    plt.show()

def calculate_rocking_scan(images, incidence_angles, sum_angle, mode='specular'):
    """
    Calculate rocking scans for the given images and incidence angles.

    Args:
        images (list): A list of calibrated 2D GISAXS images.
        incidence_angles (list): A list of corresponding incidence angles.
        sum_angle (float): The fixed summation of the incidence and exit angles.
        mode (str, optional): 'specular' for specular intensity, 'off-specular' for off-specular intensity. Defaults to 'specular'.

    Returns:
        numpy.ndarray: Rocking scan intensity values.
    """
    # Replace this with the actual method to calculate rocking scans for the images
    rocking_scan_intensity = np.array([np.sum(image) for image in images])

    return rocking_scan_intensity

def plot_rocking_scan(incidence_angles, rocking_scan_intensity, mode='specular'):
    """
    Plot the rocking scan intensity as a function of the incidence angle.

    Args:
        incidence_angles (numpy.ndarray): Incidence angles.
        rocking_scan_intensity (numpy.ndarray): Rocking scan intensity values.
        mode (str, optional): 'specular' for specular intensity, 'off-specular' for off-specular intensity. Defaults to 'specular'.
    """
    plt.figure()
    plt.plot(incidence_angles, rocking_scan_intensity)
    plt.xlabel('Incidence Angle')
    if mode == 'specular':
        plt.ylabel('Specular Intensity')
        plt.title('Rocking Scan: Specular Intensity vs Incidence Angle')
    elif mode == 'off-specular':
        plt.ylabel('Off-specular Intensity')
        plt.title('Rocking Scan: Off-specular Intensity vs Incidence Angle')
    plt.show()
