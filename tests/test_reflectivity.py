# tests/test_reflectivity.py

import unittest
import numpy as np
from xray_scatter_py import reflectivity

class TestReflectivity(unittest.TestCase):

    def setUp(self):
        self.incidence_angles = np.linspace(0.1, 2, 10)
        self.images = [np.random.random((100, 100)) for _ in range(len(self.incidence_angles))]
        self.exit_angle = 1.0
        self.sum_angle = 3.0

    def test_extract_specular_reflectivity(self):
        specular_reflectivity = reflectivity.extract_specular_reflectivity(self.images, self.incidence_angles)
        
        # Test if the output has the correct length
        self.assertEqual(len(specular_reflectivity), len(self.incidence_angles))

    def test_extract_off_specular_scattering(self):
        off_specular_scattering = reflectivity.extract_off_specular_scattering(self.images, self.incidence_angles, self.exit_angle)
        
        # Test if the output has the correct length
        self.assertEqual(len(off_specular_scattering), len(self.incidence_angles))

    def test_calculate_rocking_scan(self):
        rocking_scan_intensity_specular = reflectivity.calculate_rocking_scan(self.images, self.incidence_angles, self.sum_angle, mode='specular')
        rocking_scan_intensity_off_specular = reflectivity.calculate_rocking_scan(self.images, self.incidence_angles, self.sum_angle, mode='off-specular')
        
        # Test if the output has the correct length
        self.assertEqual(len(rocking_scan_intensity_specular), len(self.incidence_angles))
        self.assertEqual(len(rocking_scan_intensity_off_specular), len(self.incidence_angles))

if __name__ == "__main__":
    unittest.main()
