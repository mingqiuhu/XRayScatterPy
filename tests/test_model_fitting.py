# tests/test_model_fitting.py

import unittest
import numpy as np
from xray_scatter_py import model_fitting

class TestModelFitting(unittest.TestCase):

    def setUp(self):
        x = np.linspace(0, 4, 50)
        a, b, c = 2.5, 1.3, 0.5
        noise = 0.2 * np.random.normal(size=x.size)
        y = model_fitting.example_model_function(x, a, b, c) + noise
        self.data = (x, y)
        self.initial_parameters = (1, 1, 1)

    def test_example_model_function(self):
        x = np.linspace(0, 4, 50)
        y = model_fitting.example_model_function(x, 2.5, 1.3, 0.5)

        # Test if the output of the example model function has the expected shape
        self.assertEqual(y.shape, (50,))

    def test_fit_model(self):
        parameters, _ = model_fitting.fit_model(self.data, model_fitting.example_model_function, self.initial_parameters)

        # Test if the optimized parameters are close to the true values
        self.assertAlmostEqual(parameters[0], 2.5, delta=0.5)
        self.assertAlmostEqual(parameters[1], 1.3, delta=0.5)
        self.assertAlmostEqual(parameters[2], 0.5, delta=0.5)

if __name__ == "__main__":
    unittest.main()
