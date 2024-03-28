"""
DiseaseModel.py

This module provides the DiseaseModel class, which extends the BaseVisualAttention class to model
visual attention for specific diseases. It allows for the use of theoretical and empirical weights
to compute saliency maps and run experiments.

The DiseaseModel class provides methods to set active weights, compute saliency maps, run experiments
with theoretical and empirical weights, and save the results.
"""

import datetime
import os
import cv2
import numpy as np

from base_model.BaseVisualAttention import BaseVisualAttention

class DiseaseModel(BaseVisualAttention):
    """
    The DiseaseModel class extends the BaseVisualAttention class to model visual attention for specific diseases.
    It allows for the use of theoretical and empirical weights to compute saliency maps and run experiments.

    Attributes:
        theoretical_weights (dict): A dictionary containing the theoretical weights based on literature.
        empirical_weights (dict): A dictionary containing the empirical weights from trial and error.
    """
    def __init__(self, theoretical_weights=None, empirical_weights=None, parameters=None):
        """
        Initializes the DiseaseModel with disease-specific parameters.
        
        Args:
            theoretical_weights (dict, optional): A dictionary containing the theoretical weights based on literature.
            empirical_weights (dict, optional): A dictionary containing the empirical weights from trial and error.
            parameters (dict, optional): Additional model parameters, which can override the default settings.
        """
        super().__init__(parameters)

        # Use the parent's default parameters if no specific weights are provided.
        self.theoretical_weights = theoretical_weights if theoretical_weights is not None else self.parameters
        self.empirical_weights = empirical_weights if empirical_weights is not None else self.parameters


    def set_active_weights(self, weight_type='theoretical'):
        """
        Sets the active weight set for the model based on the experiment type.

        Args:
            weight_type (str, optional): A string indicating which set of weights to use ('theoretical' or 'empirical').
                Defaults to 'theoretical'.
        """
        if weight_type == 'theoretical':
            self.parameters['intensity_weight'] = self.theoretical_weights['intensity_weight']
            self.parameters['color_weight'] = self.theoretical_weights['color_weight']
            self.parameters['orientation_weight'] = self.theoretical_weights['orientation_weight']
        else:
            self.parameters['intensity_weight'] = self.empirical_weights['intensity_weight']
            self.parameters['color_weight'] = self.empirical_weights['color_weight']
            self.parameters['orientation_weight'] = self.empirical_weights['orientation_weight']

    def compute_saliency(self, image, weights):
        """
        Computes the saliency map for the given image using the specified weights.

        Args:
            image (numpy.ndarray): The input image.
            weights (dict): The weights to use for computing the saliency map.

        Returns:
            numpy.ndarray: The computed saliency map.
        """
        preprocessed_image = self.preprocess_image(image)
        self.intensity_features = self.extract_intensity_features(preprocessed_image)
        self.color_features = self.extract_color_features(preprocessed_image)
        self.orientation_features = self.extract_orientation_features(preprocessed_image)

        normalized_intensity = self.normalize_feature_maps(self.intensity_features)
        normalized_color = self.normalize_feature_maps(self.color_features)
        normalized_orientation = self.normalize_feature_maps(self.orientation_features)

        intensity_conspicuity = self.create_conspicuity_map(normalized_intensity)
        color_conspicuity = self.create_conspicuity_map(normalized_color)
        orientation_conspicuity = self.create_conspicuity_map(normalized_orientation)

        weighted_intensity = intensity_conspicuity * weights['intensity_weight']
        weighted_color = color_conspicuity * weights['color_weight']
        weighted_orientation = orientation_conspicuity * weights['orientation_weight']

        saliency_map = (weighted_intensity + weighted_color + weighted_orientation) / (weights['intensity_weight'] + weights['color_weight'] + weights['orientation_weight'])

        self.saliency_map = self.postprocess_saliency_map(saliency_map)
        self.saliency_map = cv2.resize(self.saliency_map, (image.shape[1], image.shape[0]))
        return self.saliency_map


    def run_theoretical_weights(self, image, disease_name, image_name):
        """
        Runs the model using the theoretical weights on a given image.

        Args:
            image (numpy.ndarray): The input image.
            disease_name (str): The name of the disease.
            image_name (str): The name of the image file.
       """
        saliency_map = self.compute_saliency(image, self.theoretical_weights)
        self.save_results(saliency_map, disease_name, 'theoretical', image_name)

    def run_empirical_weights(self, image, disease_name, image_name):
        """
        Runs the model using the empirical weights on a given image.

        Args:
            image (numpy.ndarray): The input image.
            disease_name (str): The name of the disease.
            image_name (str): The name of the image file.
        """
        saliency_map = self.compute_saliency(image, self.empirical_weights)
        self.save_results(saliency_map, disease_name, 'empirical', image_name)

    def run_full_experiment(self, image, disease_name, image_name):
        """
        Runs the full experiment using both sets of weights.

        Args:
            image (numpy.ndarray): The input image.
            disease_name (str): The name of the disease.
            image_name (str): The name of the image file.
        """
        self.run_theoretical_weights(image, disease_name, image_name)
        self.run_empirical_weights(image, disease_name, image_name)

    def save_results(self, saliency_map, disease_name, weight_type, image_name):
        """
        Saves the results of the experiment in a specified format and as a grayscale photo, including the input image name.
        
        Args:
            saliency_map (numpy.ndarray): The computed saliency map to save.
            disease_name (str): The name of the disease for directory structuring.
            weight_type (str): Specifies whether 'theoretical' or 'empirical' weights were used.
            image_name (str): The name of the original input image.

        Returns:
            str: The path to the directory where the results were saved.
        """
        formatted_image_name = os.path.splitext(os.path.basename(image_name))[0]
        results_dir = f'results/{disease_name}/{weight_type}/{formatted_image_name}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, f'{formatted_image_name}_saliency_map.npy'), saliency_map)
        normalized_saliency = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(results_dir, f'{formatted_image_name}_saliency_map.png'), normalized_saliency)