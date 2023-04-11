import numpy as np
from xray_scatter_py import penetration, nist

"""
Incoherent scattering is isotropic and in a small-angle scattering experiment and thus 
contributes to the background signal and degrades signal to noise
"""


# x-ray Si (density = 2.33)
sld_si = nist.get_scattering_parsed("Si", "2.33")
penetration.analyze_symmetric_penetration(1000, 1.542, sld_si['x_real'], sld_si['x_imag'], [r'$\mathrm{X-ray\ (Cu\ K_{\alpha})\ on\ Si}$'],
                                          0.2, 10, 2e+4, 0.2, 0.2)
# neutron Si (density = 2.33)
penetration.analyze_symmetric_penetration(1000, 7, sld_si['neutron_real'], sld_si['neutron_incoh'], [r'$\mathrm{Neutron\ on\ Si}$'],
                                          0.2, 10, 2e+5, 0.2, 0.2)

# neutron PSd8 (density = 1.12)
sld_psd8 = nist.get_scattering_parsed("C8D8", "1.12", neutron_wavelength="7 Ang")
penetration.analyze_symmetric_penetration(1000, 7, sld_psd8['neutron_real'], sld_psd8['neutron_incoh'], [r'$\mathrm{Neutron\ on\ PSd8}$'],
                                          0.1, 10, 1e+5, 0.05, 0.05)

# neutron PGM (density = 1.127 C7H12O4)
sld_pgm = nist.get_scattering_parsed("C7H12O4", "1.127", neutron_wavelength="7 Ang")
penetration.analyze_symmetric_penetration(1000, 7, sld_pgm['neutron_real'], sld_pgm['neutron_incoh'], [r'$\mathrm{Neutron\ on\ PGM}$'],
                                          0.1, 10, 1e+3, 0.05, 0.05)

# neutron PSd8_PGM (average)
sld_neutron_real = (sld_psd8['neutron_real'] + sld_pgm['neutron_real']) / 2
sld_neutron_incoh = (sld_psd8['neutron_incoh'] + sld_pgm['neutron_incoh']) / 2
penetration.analyze_symmetric_penetration(1000, 10.6, sld_neutron_real, sld_neutron_incoh,
                                          [r'$\mathrm{Neutron\ on\ PSd8-b-PGM}$'],
                                          0.2, 10, 1e+3, 0.1, 0.1)
penetration.analyze_assymetric_penetration(1000, 10.6, sld_neutron_real, sld_neutron_incoh,
                                           np.asarray((0.57, 1.37, 2.17)),
                                           [r'$\alpha_\mathrm{i}\ =\ 0.57^\circ$', r'$\alpha_\mathrm{i}\ =\ 1.37^\circ$',
                                            r'$\alpha_\mathrm{i}\ =\ 2.17^\circ$'],
                                           0.1, 50, 300)

# x-ray PSd8_PGM (average)
sld_x_real = (sld_psd8['x_real'] + sld_pgm['x_real']) / 2
sld_x_imag = (sld_psd8['x_imag'] + sld_pgm['x_imag']) / 2
penetration.analyze_symmetric_penetration(1000, 1.542, sld_x_real, sld_x_imag,
                                          [r'$\mathrm{X-ray\ (Cu\ K_{\alpha})\ on\ PSd8-b-PGM}$'],
                                          0.2, 10, 1e+6, 0.2, 0.2)
penetration.analyze_assymetric_penetration(1000, 1.542, sld_x_real, sld_x_imag,
                                           np.asarray((0.57, 1.37, 2.17)),
                                           [r'$\alpha_\mathrm{i}\ =\ 0.57^\circ$', r'$\alpha_\mathrm{i}\ =\ 1.37^\circ$',
                                            r'$\alpha_\mathrm{i}\ =\ 2.17^\circ$'],
                                           0.2, 50, 1e+6)
