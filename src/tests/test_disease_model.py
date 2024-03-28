import datetime
import logging
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
        
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Assert that the results are saved correctly
        theoretical_dir = f'results/TestDisease/theoretical/test_image/{timestamp}'
        logging.info(f"Checking directory: {theoretical_dir}")
        self.assertTrue(os.path.exists(theoretical_dir))
        
        saliency_map_npy = os.path.join(theoretical_dir, 'test_image_saliency_map.npy')
        logging.info(f"Checking file: {saliency_map_npy}")
        self.assertTrue(os.path.isfile(saliency_map_npy))
        
        saliency_map_png = os.path.join(theoretical_dir, 'test_image_saliency_map.png')
        logging.info(f"Checking file: {saliency_map_png}")
        self.assertTrue(os.path.isfile(saliency_map_png))

    def test_run_empirical_weights(self):
        self.model.run_empirical_weights(self.image, 'TestDisease', 'test_image.png')
        
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Assert that the results are saved correctly
        empirical_dir = f'results/TestDisease/empirical/test_image/{timestamp}'
        logging.info(f"Checking directory: {empirical_dir}")
        self.assertTrue(os.path.exists(empirical_dir))
        
        saliency_map_npy = os.path.join(empirical_dir, 'test_image_saliency_map.npy')
        logging.info(f"Checking file: {saliency_map_npy}")
        self.assertTrue(os.path.isfile(saliency_map_npy))
        
        saliency_map_png = os.path.join(empirical_dir, 'test_image_saliency_map.png')
        logging.info(f"Checking file: {saliency_map_png}")
        self.assertTrue(os.path.isfile(saliency_map_png))



if __name__ == '__main__':
    unittest.main()