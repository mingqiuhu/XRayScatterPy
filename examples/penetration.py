# examples/penetration.py

import numpy as np
from xray_scatter_py import penetration, nist

"""
    Incoherent scattering is isotropic and in a small-angle scattering experiment and thus
    contributes to the background signal and degrades signal to noise
"""


# x-ray Si (density = 2.33)
sld_si = nist.get_scattering_parsed("Si", "2.33")
dict_si_xray = {
    'num': 1000,
    'wavelength': 1.542,
    'sld': sld_si['x_real'],
    'sldi': sld_si['x_imag'],
    'legend': [r'$\mathrm{X-ray\ (Cu\ K_{\alpha})\ on\ Si}$'],
    'x_max_1d': 0.2,
    'y_min_1d': 10,
    'y_max_1d': 2e+4,
    'x_max_2d': 0.2,
    'y_max_2d': 0.2}
penetration.analyze_symmetric_penetration(**dict_si_xray)
# neutron Si (density = 2.33)
dict_si_neutron = {
    'num': 1000,
    'wavelength': 7,
    'sld': sld_si['neutron_real'],
    'sldi': sld_si['neutron_incoh'],
    'legend': [r'$\mathrm{Neutron\ on\ Si}$'],
    'x_max_1d': 0.2,
    'y_min_1d': 10,
    'y_max_1d': 2e+5,
    'x_max_2d': 0.2,
    'y_max_2d': 0.2}
penetration.analyze_symmetric_penetration(**dict_si_neutron)

# neutron PSd8 (density = 1.12)

sld_psd8 = nist.get_scattering_parsed("C8D8", "1.12")
dict_psd8_neutron = {
    'num': 1000,
    'wavelength': 7,
    'sld': sld_psd8['neutron_real'],
    'sldi': sld_psd8['neutron_incoh'],
    'legend': [r'$\mathrm{Neutron\ on\ PSd8}$'],
    'x_max_1d': 0.1,
    'y_min_1d': 10,
    'y_max_1d': 1e+5,
    'x_max_2d': 0.05,
    'y_max_2d': 0.05}
penetration.analyze_symmetric_penetration(**dict_psd8_neutron)

# neutron PGM (density = 1.127 C7H12O4)
sld_pgm = nist.get_scattering_parsed("C7H12O4", "1.127")
dict_pgm_neutron = {
    'num': 1000,
    'wavelength': 7,
    'sld': sld_pgm['neutron_real'],
    'sldi': sld_pgm['neutron_incoh'],
    'legend': [r'$\mathrm{Neutron\ on\ PGM}$'],
    'x_max_1d': 0.1,
    'y_min_1d': 10,
    'y_max_1d': 1e+3,
    'x_max_2d': 0.05,
    'y_max_2d': 0.05}
penetration.analyze_symmetric_penetration(**dict_pgm_neutron)

# neutron PSd8_PGM (average)
sld_neutron_real = (sld_psd8['neutron_real'] + sld_pgm['neutron_real']) / 2
sld_neutron_incoh = (sld_psd8['neutron_incoh'] + sld_pgm['neutron_incoh']) / 2
dict_psd8_pgm_neutron = {
    'num': 1000,
    'wavelength': 10.6,
    'sld': sld_neutron_real,
    'sldi': sld_neutron_incoh,
    'legend': [r'$\mathrm{Neutron\ on\ PSd8-b-PGM}$'],
    'x_max_1d': 0.2,
    'y_min_1d': 10,
    'y_max_1d': 1e+3,
    'x_max_2d': 0.1,
    'y_max_2d': 0.1}
penetration.analyze_symmetric_penetration(**dict_psd8_pgm_neutron)

dict_psd8_neutron_asym = {
    'num': 1000,
    'wavelength': 10.6,
    'sld': sld_neutron_real,
    'sldi': sld_neutron_incoh,
    'alpha_i_array': np.asarray((0.57, 1.37, 2.17)),
    'legend': [
        r'$\alpha_\mathrm{i}\ =\ 0.57^\circ$',
        r'$\alpha_\mathrm{i}\ =\ 1.37^\circ$',
        r'$\alpha_\mathrm{i}\ =\ 2.17^\circ$'],
    'x_max_1d': 0.1,
    'y_min_1d': 50,
    'y_max_1d': 300}
penetration.analyze_assymetric_penetration(**dict_psd8_neutron_asym)

# x-ray PSd8_PGM (average)
sld_x_real = (sld_psd8['x_real'] + sld_pgm['x_real']) / 2
sld_x_imag = (sld_psd8['x_imag'] + sld_pgm['x_imag']) / 2
dict_psd8_pgm_xray = {
    'num': 1000,
    'wavelength': 1.542,
    'sld': sld_x_real,
    'sldi': sld_x_imag,
    'legend': [r'$\mathrm{X-ray\ (Cu\ K_{\alpha})\ on\ PSd8-b-PGM}$'],
    'x_max_1d': 0.2,
    'y_min_1d': 10,
    'y_max_1d': 1e+6,
    'x_max_2d': 0.2,
    'y_max_2d': 0.2}
penetration.analyze_symmetric_penetration(**dict_psd8_pgm_xray)

dict_psd8_xray_asym = {
    'num': 1000,
    'wavelength': 1.542,
    'sld': sld_x_real,
    'sldi': sld_x_imag,
    'alpha_i_array': np.asarray((0.57, 1.37, 2.17)),
    'legend': [
        r'$\alpha_\mathrm{i}\ =\ 0.57^\circ$',
        r'$\alpha_\mathrm{i}\ =\ 1.37^\circ$',
        r'$\alpha_\mathrm{i}\ =\ 2.17^\circ$'],
    'x_max_1d': 0.2,
    'y_min_1d': 50,
    'y_max_1d': 1e+6}
penetration.analyze_assymetric_penetration(**dict_psd8_xray_asym)
