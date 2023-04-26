from xray_scatter_py import nist

sld_dict = nist.get_scattering_parsed("Si", "2.33", neutron_wavelength="7 Ang")
for key in sld_dict:
    print(key, sld_dict[key])
