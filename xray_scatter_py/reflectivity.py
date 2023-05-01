# xray_scatter_py/reflectivity.py
# xray_scatter_py/calibration.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""Calculate the reflectivity of the x-ray from a series of 2D GISAXS images
at different incidence angles.

The main functions in this module are:

calculate_relative_reflectivity: Calculate the reflectivity without normalizing
    with the incidence beam intensity.
calculate_normalized_reflectivity: Calculate the reflectivity normalized with
    the incidence beam intensity.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad


def calculate_relative_reflectivity(
        qy_fwhm: float,
        interval_degree: float,
        qy_array: np.ndarray,
        qz_array: np.ndarray,
        params: list[dict],
        images: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Calculate the reflectivity from a series of 2D GISAXS images without
    normalizing with the incidence beam intensity.

    Args:
        - qy_fwhm (float): The full width at half maximum around qy = 0.
        - interval_degree (float): The interval of incidence angles in a series
            of measurements.
        - qy_array (np.ndarray): 3D array containing qy of each detector pixel.
            The first index is the serial number of measurement.
        - qz_array (np.ndarray): 3D array containing qz of each detector pixel.
        - params (list[dict]): Each dict contains parameters of a measurement.
            Each dictionary must contain the following keys with string values:
                - 'wavelength': The wavelength of the x-ray.
                - 'sample_angle1': The incidence angle of the x-ray in degree.
                - 'det_exposure_time': The exposure time of the measurement.
        - images (np.ndarray): A 3D array of the original detector images.
            The first index is the serial number of measurement.

    Returns:
        - qz_1d (np.ndarray): 1D array containing the qz of each measurement.
        - reflectivity_array (np.ndarray): 1D array containing the reflectivity
            at each incidence angle.
        - spillover_array (np.ndarray): 1D array containing the spillover
            intensity around q = 0 at each incidence angle.
        - total_array (np.ndarray): 1D array containing the total intensity on
            the 2D detector at each incidence angle.
    """
    interval_degree = np.radians(interval_degree)
    qz_1d = np.empty(images.shape[0])
    reflectivity_array = np.empty(images.shape[0])
    spillover_array = np.empty(images.shape[0])
    total_array = np.empty(images.shape[0])
    for i in range(images.shape[0]):
        wavelength = float(params[0]['wavelength'])
        incidence_degree = np.radians(
            float(params[i]['sample_angle1']))
        time = float(params[i]['det_exposure_time'])

        qz_upper = 4 * np.pi * \
            np.sin(incidence_degree + 0.25 * interval_degree) / wavelength
        qz_lower = 4 * np.pi * \
            np.sin(incidence_degree - 0.25 * interval_degree) / wavelength
        qz_bool = np.logical_and(
            (qz_array[i]**2 + qy_array[i]**2) >= qz_lower**2,
            (qz_array[i]**2 + qy_array[i]**2) <= qz_upper**2)
        qy_bool = (np.abs(qy_array[i]) <= qy_fwhm)
        image_bool = (images[i] != -1)
        q_center_bool = ((qz_array[i]**2 + qy_array[i]**2) <= 0.0052**2)

        qz_1d[i] = 4 * np.pi * np.sin(incidence_degree) / wavelength
        reflectivity_array[i] = np.sum(
            images[i] * qz_bool * qy_bool * image_bool) / time
        spillover_array[i] = np.sum(
            images[i] * image_bool * q_center_bool) / time
        total_array[i] = np.sum(images[i] * image_bool) / time

    return qz_1d, reflectivity_array, spillover_array, total_array


# This function has some magic constants that need to be changed
def calculate_normalized_reflectivity(
        params: list[dict],
        qz_1d: np.ndarray,
        reflectivity_array: np.ndarray,
        spillover_array: np.ndarray,
        fixed_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the reflectivity from a series of 2D GISAXS images normalized
    with the incidence beam intensity.

    Args:
        - params (list[dict]): Each dict contains parameters of a measurement.
            Each dictionary must contain the following keys with string values:
                - 'wavelength': The wavelength of the x-ray.
        - qz_1d (np.ndarray): 1D array containing the qz of each measurement.
        - reflectivity_array (np.ndarray): 1D array containing the relative
            reflectivity at each incidence angle.
        - spillover_array (np.ndarray): 1D array containing the spillover
            intensity around q = 0 at each incidence angle.
        - fixed_sigma (float): The standard deviation of the beam divergence.

    Returns:
        - normalized_reflectivity (np.ndarray): 1D array containing the
            normalized reflectivity at each incidence angle.
        - fitted_spillover (np.ndarray): 1D array containing the spillover
            intensity around q = 0 predicted by the fitting results.
    """
    wavelength = float(params[0]['wavelength'])
    theta_1d = np.arcsin(qz_1d * wavelength / 4 / np.pi)

    # Define the integrand function to calculate the beam intensity on sample
    def integrand(y_sample, incidence_intensity):
        return incidence_intensity /\
            (fixed_sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * y_sample ** 2 /
                                                        fixed_sigma ** 2)

    # Define the function to fit
    def spillover_func(theta, incidence_intensity, len_sample):
        fitted_spillover = []
        for t in theta:
            integral, _ = quad(integrand, 0, len_sample *
                               np.sin(t) / 2, args=(incidence_intensity,))
            fitted_spillover.append(0.5 * incidence_intensity - integral)
        return np.array(fitted_spillover)

    # Fit the integrated spillover intensity to experimental data
    # Initial guess for incidence_intensity, len_sample
    initial_guess = [2.5e6, 10]
    optimal_params, _ = curve_fit(
        spillover_func, theta_1d[:-5], spillover_array[:-5], p0=initial_guess)
    incidence_intensity, len_sample = optimal_params
    print("Optimal I0:", incidence_intensity)
    print("Optimal L0:", len_sample)

    fitted_spillover = []
    beam_on_sample = []
    for t in theta_1d:
        integral, _ = quad(integrand, 0, len_sample *
                           np.sin(t) / 2, args=(incidence_intensity,))
        fitted_spillover.append(0.5 * incidence_intensity - integral)
        beam_on_sample.append(2 * integral)
    normalized_reflectivity = reflectivity_array / np.array(beam_on_sample)
    return normalized_reflectivity, np.array(fitted_spillover)
