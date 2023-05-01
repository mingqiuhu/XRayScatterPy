# tests/test_data_processing.py

import unittest
import numpy as np
from xray_scatter_py import data_processing


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.data_2D = np.random.random((100, 100))

    def test_process_data(self):
        # Replace with actual file paths and calibration parameters
        raw_data_path = "path/to/raw_data.tif"
        q_calibration = {"param1": "value1", "param2": "value2"}
        intensity_calibration = {"param1": "value1", "param2": "value2"}

        processed_data = data_processing.process_data(
            raw_data_path, q_calibration, intensity_calibration)

        # Test if processed data has the expected shape
        self.assertEqual(processed_data.shape, (100, 100))
        # Add more tests as needed

    def test_integrate_1D_q(self):
        integrated_data = data_processing.integrate_1D_q(self.data_2D)

        # Test if integrated data has the expected shape
        self.assertEqual(integrated_data.shape, (100,))
        # Add more tests as needed

    def test_calculate_beam_divergence(self):
        beam_divergence = data_processing.calculate_beam_divergence(
            self.data_2D)

        # Test if beam_divergence is a float
        self.assertIsInstance(beam_divergence, float)
        # Add more tests as needed


if __name__ == "__main__":
    unittest.main()
