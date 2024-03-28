import unittest
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.ControlModel import ControlModel, run_control_experiment

class TestControlModel(unittest.TestCase):
    def setUp(self):
        self.model = ControlModel()
        self.image_path = './tests/test_image.png'

    def test_initialization(self):
        self.assertIsNotNone(self.model.parameters)

    def test_run_control_experiment(self):
        run_control_experiment(self.image_path, 'tests/control_results')
        # Assert that the results are saved correctly

if __name__ == '__main__':
    unittest.main()