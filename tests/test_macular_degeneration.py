import unittest
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.MacularDegeneration import MacularDegenerationModel, run_macular_degeneration_experiment

class TestMacularDegeneration(unittest.TestCase):
    def setUp(self):
        self.model = MacularDegenerationModel()
        self.image_path = 'tests/test_image.png'

    def test_initialization(self):
        self.assertIsNotNone(self.model.theoretical_weights)
        self.assertIsNotNone(self.model.empirical_weights)

    def test_run_macular_degeneration_experiment(self):
        run_macular_degeneration_experiment(self.image_path, 'tests/macular_degeneration_results')
        # Assert that the results are saved correctly

if __name__ == '__main__':
    unittest.main()