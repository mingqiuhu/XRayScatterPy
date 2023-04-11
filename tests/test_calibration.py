import unittest
import os
import numpy as np
from xray_scatter_py import calibration, utils

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiff_files_calibration_q')
START_INDEX = 77023
END_INDEX = 77026

class TestCalculateAngleAndQ(unittest.TestCase):

    def setUp(self):
        self.detx0 = 100.4
        self.params_dict_list, self.image_array = utils.read_multiimage(DATA_PATH, START_INDEX, END_INDEX)

    def test_calculate_angle(self):
        theta_array, azimuth_array = calibration.calculate_angle(self.detx0, self.params_dict_list, self.image_array)

        # Check if the returned arrays have the same shape as the input image array
        self.assertEqual(theta_array.shape, self.image_array.shape)
        self.assertEqual(azimuth_array.shape, self.image_array.shape)

        # Check if the returned arrays contain finite values
        self.assertTrue(np.all(np.isfinite(theta_array)))
        self.assertTrue(np.all(np.isfinite(azimuth_array)))


    def test_calculate_q(self):
        qx_array, qy_array, qz_array = calibration.calculate_q(self.detx0, self.params_dict_list, self.image_array)

        # Check if the returned arrays have the same shape as the input image array
        self.assertEqual(qx_array.shape, self.image_array.shape)
        self.assertEqual(qy_array.shape, self.image_array.shape)
        self.assertEqual(qz_array.shape, self.image_array.shape)

        # Check if the returned arrays contain finite values
        self.assertTrue(np.all(np.isfinite(qx_array)))
        self.assertTrue(np.all(np.isfinite(qy_array)))
        self.assertTrue(np.all(np.isfinite(qz_array)))


if __name__ == '__main__':
    unittest.main()
