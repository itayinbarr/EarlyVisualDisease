import unittest
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base_model.DiseaseModel import DiseaseModel

class TestDiseaseModel(unittest.TestCase):
    def setUp(self):
        self.theoretical_weights = {'intensity_weight': 0.5, 'color_weight': 0.3, 'orientation_weight': 0.2}
        self.empirical_weights = {'intensity_weight': 0.4, 'color_weight': 0.4, 'orientation_weight': 0.2}
        self.model = DiseaseModel(self.theoretical_weights, self.empirical_weights)
        self.image = cv2.imread('tests/test_image.png')

    def test_initialization(self):
        self.assertEqual(self.model.theoretical_weights, self.theoretical_weights)
        self.assertEqual(self.model.empirical_weights, self.empirical_weights)

    def test_run_theoretical_weights(self):
        self.model.run_theoretical_weights(self.image, 'TestDisease', 'test_image.png')
        # Assert that the results are saved correctly

    def test_run_empirical_weights(self):
        self.model.run_empirical_weights(self.image, 'TestDisease', 'test_image.png')
        # Assert that the results are saved correctly

    def test_run_full_experiment(self):
        self.model.run_full_experiment(self.image, 'TestDisease', 'test_image.png')
        # Assert that the results are saved correctly

if __name__ == '__main__':
    unittest.main()