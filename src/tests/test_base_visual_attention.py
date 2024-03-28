
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# tests/test_base_visual_attention.py
import unittest
import cv2
import numpy as np
from base_model.BaseVisualAttention import BaseVisualAttention

class TestBaseVisualAttention(unittest.TestCase):
    def setUp(self):
        self.model = BaseVisualAttention()
        self.image = cv2.imread('./tests/test_image.png')
        if self.image is None:
            raise Exception("Failed to load the test image. Please check the file path and ensure the image exists.")

    def test_initialization(self):
        self.assertIsNotNone(self.model.parameters)

    def test_compute_saliency(self):
        saliency_map = self.model.compute_saliency(self.image)
        self.assertIsNotNone(saliency_map)
        self.assertEqual(saliency_map.shape, (self.image.shape[0], self.image.shape[1]))

    def test_preprocess_image(self):
        preprocessed_image = self.model.preprocess_image(self.image)
        self.assertEqual(preprocessed_image.shape, (1088, 1983, 3))

    def test_extract_intensity_features(self):
        preprocessed_image = self.model.preprocess_image(self.image)
        intensity_features = self.model.extract_intensity_features(preprocessed_image)
        self.assertEqual(len(intensity_features), 6)

    def test_extract_color_features(self):
        preprocessed_image = self.model.preprocess_image(self.image)
        color_features = self.model.extract_color_features(preprocessed_image)
        self.assertEqual(len(color_features), 12)

    def test_extract_orientation_features(self):
        preprocessed_image = self.model.preprocess_image(self.image)
        orientation_features = self.model.extract_orientation_features(preprocessed_image)
        self.assertEqual(len(orientation_features), 12)

    def test_create_gaussian_pyramid(self):
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pyramid = self.model.create_gaussian_pyramid(grayscale_image, levels=5)
        self.assertEqual(len(pyramid), 5)

    def test_center_surround_diff(self):
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pyramid = self.model.create_gaussian_pyramid(grayscale_image, levels=5)
        center_surround_diff = self.model.center_surround_diff(pyramid, 2, 4)
        self.assertIsNotNone(center_surround_diff)

    def test_normalize_feature_maps(self):
        preprocessed_image = self.model.preprocess_image(self.image)
        intensity_features = self.model.extract_intensity_features(preprocessed_image)
        normalized_maps = self.model.normalize_feature_maps(intensity_features)
        self.assertEqual(len(normalized_maps), len(intensity_features))

    def test_create_conspicuity_map(self):
        preprocessed_image = self.model.preprocess_image(self.image)
        intensity_features = self.model.extract_intensity_features(preprocessed_image)
        normalized_maps = self.model.normalize_feature_maps(intensity_features)
        conspicuity_map = self.model.create_conspicuity_map(normalized_maps)
        self.assertIsNotNone(conspicuity_map)

    def test_postprocess_saliency_map(self):
        saliency_map = np.random.rand(100, 100)
        postprocessed_map = self.model.postprocess_saliency_map(saliency_map)
        self.assertIsNotNone(postprocessed_map)

    def test_N(self):
        image = np.random.rand(100, 100)
        normalized_image = self.model.N(image)
        self.assertIsNotNone(normalized_image)

    def test_apply_inhibition_of_return(self):
        # Create a random saliency map
        saliency_map = np.random.rand(100, 100)
        
        # Define the winner location
        winner_location = (50, 50)
        
        # Create a copy of the original saliency map
        original_saliency_map = saliency_map.copy()
        
        # Apply inhibition of return
        self.model.apply_inhibition_of_return(saliency_map, winner_location)
        
        # Assert that the saliency map has been updated correctly
        inhibition_radius = self.model.parameters.get('inhibition_radius', 50)
        inhibition_strength = self.model.parameters.get('inhibition_strength', 0.2)
        
        # Create a circular mask for the inhibition region
        height, width = saliency_map.shape
        y, x = winner_location
        y_grid, x_grid = np.ogrid[-y:height-y, -x:width-x]
        mask = x_grid**2 + y_grid**2 <= inhibition_radius**2
        
        # Assert that the saliency values within the inhibition region have been reduced
        self.assertTrue(np.all(saliency_map[mask] <= original_saliency_map[mask] * (1 - inhibition_strength)))
        
        # Assert that the saliency values outside the inhibition region remain unchanged
        self.assertTrue(np.all(saliency_map[~mask] == original_saliency_map[~mask]))
        
if __name__ == '__main__':
    unittest.main()