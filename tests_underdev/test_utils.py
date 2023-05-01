# tests/test_utils.py
import os
import unittest
import numpy as np
from xray_scatter_py import utils


GRAD_FILE = '06022022 AgBH ESAXS.grad'

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'tiff_files_calibration_q')
START_INDEX = 77023
END_INDEX = 77026


class TestImageFileFunctions(unittest.TestCase):

    def test_read_image(self):
        params_dict, image = utils.read_image(DATA_PATH, START_INDEX)

        self.assertIsInstance(params_dict, dict)
        self.assertGreater(len(params_dict), 0)
        self.assertTrue('detx' in params_dict)

        self.assertIsInstance(image, np.ndarray)
        self.assertGreater(len(image.shape), 0)
        self.assertTrue(image.shape == (619, 487))

    def test_read_multiimage(self):
        params_dict_list, image_array = utils.read_multiimage(
            DATA_PATH, START_INDEX, END_INDEX)

        self.assertIsInstance(params_dict_list, list)
        self.assertEqual(len(params_dict_list), END_INDEX - START_INDEX + 1)
        for params_dict in params_dict_list:
            self.assertIsInstance(params_dict, dict)
            self.assertGreater(len(params_dict), 0)
            self.assertTrue('detx' in params_dict)

        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(image_array.shape[0], END_INDEX - START_INDEX + 1)
        self.assertTrue(image_array[0].shape == (619, 487))


class TestReadGradFile(unittest.TestCase):

    def test_read_grad_file(self):
        header_info, data_array, xml_dict = utils.read_grad_file(
            DATA_PATH, GRAD_FILE)

        self.assertIsInstance(header_info, dict)
        self.assertGreater(len(header_info), 0)

        self.assertIsInstance(data_array, np.ndarray)
        self.assertGreater(data_array.shape[0], 0)

        self.assertIsInstance(xml_dict, dict)
        self.assertGreater(len(xml_dict), 0)
        self.assertTrue('detx' in xml_dict)


"""
    class
class TestUtils(unittest.TestCase):

    def_test_read_image_file(self):
        directory_path = ''

    def test_q_to_qz_qy(self):
        q_values = np.array([0.1, 0.2, 0.3])
        q_angle = np.pi / 4
        qz_qy_values = utils.q_to_qz_qy(q_values, q_angle)

        # Test if the output has the correct length
        self.assertEqual(len(qz_qy_values), len(q_values))

    def test_rad_to_deg(self):
        radians = np.array([0, np.pi / 2, np.pi])
        degrees = utils.rad_to_deg(radians)

        # Test if the output has the correct values
        np.testing.assert_array_almost_equal(degrees, [0, 90, 180])

    def test_deg_to_rad(self):
        degrees = np.array([0, 90, 180])
        radians = utils.deg_to_rad(degrees)

        # Test if the output has the correct values
        np.testing.assert_array_almost_equal(radians, [0, np.pi / 2, np.pi])
"""
if __name__ == "__main__":
    unittest.main()
