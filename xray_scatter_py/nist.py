# -*- coding: utf-8 -*-
# xray_scatter_py/nist.py
# Authors: Mingqiu Hu, Xuchen Gan in Prof. Thomas P. Russell's group
# This package is developed using Umass Amherst central facility resources.
"""Visity the website of naitonal institute of standards and technology (NIST)
to get the scattering length density of a material.

The main functions in this module are used to:

get_scattering_json: get scattering length density of a material from NIST
get_scattering_parsed: get the parsed scattering length density
"""

import requests
import json


def get_scattering_json(material: str,
                        density: str,
                        neutron_wavelength: str = "7 Ang") -> str:
    """Get the scattering length density of a material from NIST.

    Args:
        - material (str): chemical formula of the material.
        - density (str): density of the material in g/cm^3.
        - neutron_wavelength (str, optional): wavelength of the neutron in Å.
            If not specified, the default value is 7 Å.

    Returns:
        - str: a json string containing the scattering length density of the
            material.
    """
    url = "https://www.ncnr.nist.gov/cgi-bin/nact.py"

    form_data = {
        "calculate": "scattering",
        "sample": material,
        "flux": "1e8",
        "fast": "0",
        "Cd": "0",
        "mass": "",
        "density": density,
        "thickness": "1",
        "wavelength": neutron_wavelength,
        "xray": "Cu Ka",
        "exposure": "10",
        "rest[]": "0",
        "rest[]": "1",
        "rest[]": "24",
        "rest[]": "360",
        "rest[]": "1 y",
        "decay": "0.0005",
        "abundance": "IAEA"
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/"
                       "537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/"
                       "537.36"),
        "Referer": "https://www.ncnr.nist.gov/resources/activation/",
        "Origin": "https://www.ncnr.nist.gov",
    }

    response = requests.post(url, data=form_data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error submitting form. Status code: {response.status_code}")
        return None


def get_scattering_parsed(material: str,
                          density: str,
                          neutron_wavelength: str = "7 Ang") -> dict:
    """Get the scattering length density of a material from NIST.
        The json string is parsed into a dictionary.

    Args:
        - material (str): chemical formula of the material.
        - density (str): density of the material in g/cm^3.
        - neutron_wavelength (str, optional): wavelength of the neutron in Å.
            If not specified, the default value is 7 Å.

    Returns:
        - dict: a dictionary containing the scattering length density.
    """
    json_scattering = get_scattering_json(
        material, density, neutron_wavelength=neutron_wavelength)
    dict_scattering = json.loads(json_scattering)
    neutron_real = dict_scattering["scattering"]["sld"]["real"]
    neutron_imag = dict_scattering["scattering"]["sld"]["imag"]
    neutron_incoh = dict_scattering["scattering"]["sld"]["incoh"]
    x_real = dict_scattering["xray_scattering"]["sld"]["real"]
    x_imag = dict_scattering["xray_scattering"]["sld"]["imag"]
    return {"neutron_real": neutron_real * 1e-6,
            "neutron_imag": neutron_imag * 1e-6,
            "neutron_incoh": neutron_incoh * 1e-6,
            "x_real": x_real * 1e-6,
            "x_imag": x_imag * 1e-6}
