# tests/test_gratings.py

import unittest
import numpy as np
from xray_scatter_py import gratings

class TestGratings(unittest.TestCase):

    def test_incidence_angle(self):
        grating_direction = np.array([1, 0, 0])
        xray_direction = np.array([0, 1, 0])
        angle = gratings.incidence_angle(grating_direction, xray_direction)

        # Test if the output has the correct angle
        self.assertAlmostEqual(angle, np.pi / 2)

    def test_ewald_sphere(self):
        wavelength = 1.54
        detector_distance = 1000
        detector_size = (256, 256)
        ewald_sphere_result = gratings.ewald_sphere(wavelength, detector_distance, detector_size)

        # Test if the output has the correct shape
        self.assertEqual(ewald_sphere_result.shape, (*detector_size, 3))

    def test_reciprocal_space_lattice(self):
        grating_parameters = {'param1': 1, 'param2': 2}
        reciprocal_space_lattice_result = gratings.reciprocal_space_lattice(grating_parameters)

        # Test if the output has the correct shape
        self.assertEqual(reciprocal_space_lattice_result.shape, (3, 3))

    def test_scattering_pattern_location(self):
        ewald_sphere = np.zeros((256, 256, 3))
        reciprocal_space_lattice = np.zeros((3, 3))
        location = gratings.scattering_pattern_location(ewald_sphere, reciprocal_space_lattice)

        # Test if the output has the correct length
        self.assertEqual(len(location), 2)

if __name__ == "__main__":
    unittest.main()
