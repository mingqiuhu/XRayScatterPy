# -*- coding: utf-8 -*-
# xray_scatter_py/utils.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""Utility functions for reeading, writing, and type checking.

This module provides functions to check the shape of numpy arrays, and to read
the original experimental data, and to write the results to files.

The main functions in this module are used to:

validate_kwargs: Assert that all keyword arguments are valid.
validate_array_dimension: Assert that the given array has expected dimension.
validate_list_len: Assert that the given list has expected length.
validate_array_shape: Assert that all given arrays have the same shape.
read_image: Read a TIFF image file of GANESHA SAXSLAB and its metadata from.
read_synchrotron_image: Read a TIFF image file from APS Berkeley.
read_multiimage: Read multiple TIFF image files and their metadata.
read_grad_file: Read an exported 1D data from GANESHA SAXSLAB.
read_sans: Read a neutron scattering data file from Oak Ridge National Lab.
debye_scherrer_xray: Calculate the D-S ring in X-Ray scattering.
debye_scherrer_neutron: Calculate the D-S ring in neutron scattering.
"""
import os
import re
import tifffile

import numpy as np
import xmltodict


def validate_kwargs(valid_kwargs: set, kwargs: dict):
    """Assert that all keyword arguments are valid.

    Args:
        - valid_kwargs (set): A set of valid keyword arguments.
        - kwargs (dict): A dictionary of keyword arguments.
    """

    unrecognized_kwargs = set(kwargs.keys()) - valid_kwargs
    if unrecognized_kwargs:
        raise ValueError(
            f"Unrecognized keyword arguments: {unrecognized_kwargs}")


def validate_array_dimension(ndarray: np.ndarray, dimension: int):
    """Assert that the given array has the expected dimension.

    Args:
        - ndarray (np.ndarray): A NumPy array.
        - dimension (int): The expected dimension of the array.
    """
    if len(ndarray.shape) != dimension:
        raise ValueError(
            f"The given array has a dimension of {len(ndarray.shape)}, "
            f"but the expected dimension is {dimension}.")


def validate_list_len(params_dict_list: list[dict], len_: int):
    """Assert that the params_dict_list has the expected length.

    Args:
        - params_dict_list (list[dict]): A list of the experiment parameters
            of each measurement.
        - len_ (int): The expected number of total measurements.
    """
    if len(params_dict_list) != len_:
        raise ValueError(
            f"The given parameters dictionary list has a length of "
            f"{len(params_dict_list)}, "
            f"but the expected length is {len_}.")


def validate_array_shape(*args):
    """Assert that all given arrays have the same shape.

    Args:
        - *args (np.ndarray): A variable number of NumPy arrays.
    """

    shapes = [arg.shape for arg in args]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"Array shapes {shapes} are not uniform.")


def read_image(
        directory_path: str,
        index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a TIFF image file of GANESHA SAXSLAB and its metadata.

    Args:
        - directory_path (str): The directory path of the TIFF file.
        - index (int): The serial nunmber of the measurement to be read.

    Returns:
        - tuple: A tuple containing a dictionary with the parsed metadata
            parameters and the images data as 3D a NumPy array.

    Raises:
        - FileNotFoundError: If the given directory path or file index does not
            exist.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    file_template = "latest_{:07d}_craw.tiff"
    file_path = os.path.join(directory_path, file_template.format(index))

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with tifffile.TiffFile(file_path) as tiff:
        first_page = tiff.pages[0]
        image = first_page.asarray()
        tags = first_page.tags

        xml_tag = next(
            tag for tag_name, tag in tags.items() if 'xml' in str(tag.value))
        xml = xml_tag.value

    xml_dict = xmltodict.parse(xml)
    parameters = xml_dict['SAXSLABparameters']
    params_dict = {
        param['@name']: param.get('#text', None)
        for param in parameters['param']}

    return params_dict, image


def read_synchrotron_image(directory_path: str, file_name: str) -> np.ndarray:
    """Read a TIFF image file from the advanced light source in Berkeley.

    Args:
        - directory_path (str): The directory path of the TIFF file.
        - file_name (str): The name of the TIFF file.

    Returns:
        - np.ndarray: The 2D image of the measurement.

    Raises:
        - FileNotFoundError: If the given directory path or file name does not
            exist.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    file_path = os.path.join(directory_path, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with tifffile.TiffFile(file_path) as tiff:
        first_page = tiff.pages[0]
        image = first_page.asarray()
    return image


def read_multiimage(
        directory_path: str,
        start_index: int,
        end_index: int = 0) -> tuple[list[dict], np.ndarray]:
    """Read multiple TIFF images and their metadata from the given directory.

    Args:
        - directory_path (str): The directory where the TIFF files are located.
        - start_index (int): The starting index of the TIFF files to be read.
        - end_index (int, optional): The ending index of the TIFF files. If not
            provided, only the image with the start_index will be read.

    Returns:
        - tuple: A tuple containing a list of dictionaries with the parsed
            metadata parameters and a 3D NumPy array for all images.

    Raises:
        - FileNotFoundError: If the directory or file path does not exist.
    """

    num_images = end_index - start_index + 1 if end_index else 1
    first_params_dict, first_image = read_image(directory_path, start_index)

    image_shape = first_image.shape
    image_array = np.empty((num_images, *image_shape), dtype=first_image.dtype)
    image_array[0] = first_image

    params_dict_list = [first_params_dict]

    for i in range(1, num_images):
        image_index = start_index + i
        params_dict, image = read_image(directory_path, image_index)
        image_array[i] = image
        params_dict_list.append(params_dict)

    return params_dict_list, image_array


def read_grad_file(
        directory_path: str,
        file_name: str) -> tuple[dict, np.ndarray, dict]:
    """Read and parse data from an exported GRAD from GANESHA SAXSLAB.

    Args:
        - directory_path (str): The directory where the file is located.
        - file_name (str): The name of the GRAD file.

    Returns:
        tuple: A tuple containing:
            - header_info (dict): A dictionary containing the GRAD header.
            - data_array (np.ndarray): A NumPy array containing the data.
            - xml_dict (dict): A dictionary containing the parsed XML data.

    Raises:
        - FileNotFoundError: If the directory or file is not found.
    """

    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    file_path = os.path.join(directory_path, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    header_info = {}
    data_list = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        header_info.update({
            "num_datasets": int(lines[0].split(',')[0]),
            "num_col_per_dataset": int(lines[1].split(',')[0]),
            "num_rows": int(lines[2].split(',')[0])
        })

        pattern = (
            r"(?P<x>\w+)(?=-units).*?\((?P<x_unit>[^)]+)\).*?"
            r"(?P<y>\w+)(?=-units).*?\((?P<y_unit>[^)]+)\)")
        match = re.search(pattern, lines[3])
        x, y, x_unit, y_unit = match.group("x"), match.group(
            "y"), match.group("x_unit"), match.group("y_unit")

        header_info.update({
            "x": x,
            "y": y,
            "x_unit": x_unit,
            "y_unit": y_unit
        })

        file_name, measurement_description, _ = lines[4].strip(
            '"\n').split('","')
        header_info.update({
            "file_name": file_name,
            "measurement_description": measurement_description
        })

        data_start_line = 6
        data_end_line = data_start_line

        for line in lines[data_start_line:]:
            if line.startswith("#"):
                break
            data_end_line += 1
            data = list(map(float, line.split(',')))
            if not any(np.isnan(data)):
                data_list.append(data)

        data_array = np.array(data_list)

        xml_dict = xmltodict.parse(lines[data_end_line + 2])
        xml_dict = xml_dict['ROOT']

    return header_info, data_array, xml_dict


def read_sans(directory_path: str, file_name: str, header=1):
    """Read neutron scattering data from Oak Ridge National Lab.

    Args:
        - directory_path (str): The directory where the file is located.
        - file_name (str): The name of the neutron scattering data containing
            three columns, qy, qz, and intensity.
        - header (int, optional): The number of header lines to skip.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    file_path = os.path.join(directory_path, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data_array = np.genfromtxt(file_path, dtype=float, skip_header=header)

    return data_array


def wave_vector(a, wavelength):
    """Calculate wave vector for certain incidence angle and wave length.

    Args:
        - a (float): The incidence angle, in radian unit.
        - wavelength (float): The wavelength of incidence beam, in Angstrom unit.

    Return:
        - q (float): The corresponding wavevector, in Angstrom unit.
    """
    k = 2*np.pi*np.sin(a)/wavelength
    return k


def debye_scherrer_xray(qy_array, wavelength, L0, m, aiz, acp):
    """calculate the Debye Scherrer ring for X-Ray scattering.

    Args:
        - qy_array (np.ndarray): The qy value array, 1D.
        - L0 (float): The domain spacing size, in Angstrom unit.
        - m (Int): The order of scatering.
        - aiz (float): The incidence angle, in radian unit.
        - acp (float): The polymer air interface critical angle, degree.

    Returns:
        - qzu_array (np.ndarray): The qz list of upper Debye-Scherrer ring
        - qzl_array (np.ndarray): The qz list of lower Debye-Scherrer ring
    """
    kiz = wave_vector(aiz, wavelength)
    kcp = wave_vector(acp, wavelength)
    sqrt1 = (kiz**2 - kcp**2)**0.5
    qzu_array, qzl_array = qy_array, qy_array
    for i in range(0, len(qy_array)):
        sqrt2 = ((2*np.pi*m/L0)**2 - qy_array[i]**2)**0.5
        # a_i > 0, a_f > 0, reflect from polymer air interface
        qzu_array[i] = kiz + (kcp**2 + (sqrt2 + sqrt1)**2)**0.5
        # a_i > 0, a_f > 0, reflect from polymer substrate interface
        qzl_array[i] = kiz + (kcp**2 + (sqrt2 - sqrt1)**2)**0.5
    return [qzu_array, qzl_array]


def debye_scherrer_neutron(qy_array, wavelength, L0, m, aiz, acp, acs):
    """calculate the Debye Scherrer ring for neutron scattering.

    Args:
        - qy_array (np.ndarray): The qy value array, 1D.
        - L0 (float): The domain spacing size, in Angstrom unit.
        - m (Int): The order of scatering.
        - aiz (float): The incidence angle, in radian unit.
        - acp (float): The polymer air interface critical angle, radian.
        - acs (float): The substrate air interface critical angle, radian.

    Returns:
        - qzru_array (np.ndarray): qz list of upper reflection DS ring
        - qzrl_array (np.ndarray): qz list of lower reflection DS ring
        - qztu_array (np.ndarray): qz list of upper transmission DS ring
        - qztl_array (np.ndarray): qz list of lower transmission DS ring
    """
    kiz = wave_vector(aiz, wavelength)
    kcp = wave_vector(acp, wavelength)
    kcs = wave_vector(acs, wavelength)
    kcsp = (kcp**2-kcs**2)**0.5
    sqrt1 = (kiz**2 - kcp**2)**0.5
    sqrt2 = (kiz**2 - kcsp**2)**0.5
    qzru_array, qzrl_array, qztu_array, qztl_array = [
        np.zeros_like(qy_array) for _ in range(4)]
    if aiz > 0:
        for i in range(len(qy_array)):
            sqrt0 = ((2*np.pi*m/L0)**2 - (qy_array[i])**2)**0.5
            # a_i > 0, a_f > 0, upper reflection
            qzru_array[i] = kiz + (kcp**2 + (sqrt0 + sqrt1)**2)**0.5
            # a_i > 0, a_f > 0, lower reflection
            qzrl_array[i] = kiz + (kcp**2 + (sqrt0 - sqrt1)**2)**0.5
            # a_i > 0, a_f < 0, upper transmission
            qztu_array[i] = kiz - (kcsp**2 + (sqrt0 - sqrt1)**2)**0.5
            # a_i > 0, a_f < 0, lower transmission
            qztl_array[i] = kiz - (kcsp**2 + (sqrt0 + sqrt1)**2)**0.5
    if aiz <= 0:
        for i in range(len(qy_array)):
            sqrt0 = ((2*np.pi*m/L0)**2 - (qy_array[i])**2)**0.5
            # a_i > 0, a_f > 0, upper reflection
            qzru_array[i] = kiz + (kcp**2 + (sqrt0 + sqrt2)**2)**0.5
            # a_i > 0, a_f > 0, lower reflection
            qzrl_array[i] = kiz + (kcp**2 + (sqrt0 - sqrt2)**2)**0.5
            # a_i > 0, a_f < 0, upper transmission
            qztu_array[i] = kiz - (kcsp**2 + (sqrt0 - sqrt2)**2)**0.5
            # a_i > 0, a_f < 0, lower transmission
            qztl_array[i] = kiz - (kcsp**2 + (sqrt0 + sqrt2)**2)**0.5
    return [qzru_array, qzrl_array, qztu_array, qztl_array]
