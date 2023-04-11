# xray_scatter_py/utils.py

import os
import re
import numpy as np
import tifffile
import xmltodict

def read_image(directory_path: str, index: int) -> tuple:
    """
    Read a TIFF image file of GANESHA SAXSLAB and its metadata from the given directory path and index.

    Args:
        directory_path (str): The directory path where the TIFF file is located.
        index (int): The index of the TIFF file to be read.

    Returns:
        tuple: A tuple containing a dictionary with the parsed metadata parameters and the image data as a NumPy array.

    Raises:
        FileNotFoundError: If the given directory path or file index does not exist.
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

        xml_tag = next(tag for tag_name, tag in tags.items() if 'xml' in str(tag.value))
        xml = xml_tag.value

    xml_dict = xmltodict.parse(xml)
    parameters = xml_dict['SAXSLABparameters']
    params_dict = {param['@name']: param.get('#text', None) for param in parameters['param']}

    return params_dict, image


def read_multiimage(directory_path: str, start_index: int, end_index: int = 0) -> tuple:
    """
    Read multiple TIFF image files and their metadata from the given directory path and index range.

    Args:
        directory_path (str): The directory path where the TIFF files are located.
        start_index (int): The starting index of the TIFF files to be read.
        end_index (int, optional): The ending index of the TIFF files to be read. If not provided, only the image
                                   with the start_index will be read. Defaults to 0.

    Returns:
        tuple: A tuple containing a list of dictionaries with the parsed metadata parameters and a NumPy array
               with the image data for all images.

    Raises:
        FileNotFoundError: If the given directory path or file path does not exist.
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


def read_grad_file(directory_path: str, file_name: str) -> tuple:
    """
    Read and parse data from a GRAD file with given directory and file_name.

    Args:
        directory_path (str): The directory where the file is located.
        file_name (str): The name of the GRAD file.

    Returns:
        tuple: A tuple containing:
            - header_info (dict): A dictionary containing the header information.
            - data_array (np.ndarray): A NumPy array containing the data.
            - xml_dict (dict): A dictionary containing the parsed XML data.

    Raises:
        FileNotFoundError: If the directory or file is not found.
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
            "Number of Datasets": int(lines[0].split(',')[0]),
            "Number of Columns per Dataset": int(lines[1].split(',')[0]),
            "Maximum Number of Rows for Any Dataset": int(lines[2].split(',')[0])
        })

        pattern = r"(?P<x>\w+)(?=-units).*?\((?P<x_unit>[^)]+)\).*?(?P<y>\w+)(?=-units).*?\((?P<y_unit>[^)]+)\)"
        match = re.search(pattern, lines[3])
        x, y, x_unit, y_unit = match.group("x"), match.group("y"), match.group("x_unit"), match.group("y_unit")

        header_info.update({
            "x": x,
            "y": y,
            "x_unit": x_unit,
            "y_unit": y_unit
        })

        file_name, measurement_description, _ = lines[4].strip('"\n').split('","')
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

        xml_dict = xmltodict.parse(lines[data_end_line+2])
        xml_dict = xml_dict['ROOT']

    return header_info, data_array, xml_dict

def read_sans(directory_path: str, file_name: str, header=1):
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    file_path = os.path.join(directory_path, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data_array = np.genfromtxt(file_path, dtype=float, skip_header=header)

    return data_array