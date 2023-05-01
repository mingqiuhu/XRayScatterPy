# tests/test_math_utils.py

import unittest
from xray_scatter_py import math_utils


class TestMathUtils(unittest.TestCase):

    def test_custom_function_1(self):
        x = 3
        y = 4
        result = math_utils.custom_function_1(x, y)

        # Test if the output has the correct value
        self.assertEqual(result, 7)

    def test_custom_function_2(self):
        x = 2
        y = 3
        z = 1
        result = math_utils.custom_function_2(x, y, z)

        # Test if the output has the correct value
        self.assertEqual(result, 5)

    # Add more test cases for your other custom mathematical functions


if __name__ == "__main__":
    unittest.main()
