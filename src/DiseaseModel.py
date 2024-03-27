import datetime
import os

import cv2
import numpy as np
from BaseVisualAttention import BaseVisualAttention

class DiseaseModel(BaseVisualAttention):
    def __init__(self, theoretical_weights, empirical_weights, parameters=None):
        """
        Initializes the DiseaseModel with disease-specific parameters.

        :param theoretical_weights: A dictionary containing the theoretical weights based on literature.
        :param empirical_weights: A dictionary containing the empirical weights from trial and error.
        :param parameters: Additional model parameters, which can override the default settings.
        """
        # Initialize the base class with the provided parameters.
        super().__init__(parameters)

        # Store the disease-specific weights.
        self.theoretical_weights = theoretical_weights
        self.empirical_weights = empirical_weights

        # Choose which set of weights to use for the model.
        # This can be switched between 'theoretical' and 'empirical' based on your experiment's needs.
        self.active_weight_set = 'theoretical'  # Default to theoretical weights.

    def set_active_weights(self, weight_type='theoretical'):
        """
        Sets the active weight set for the model based on the experiment type.

        :param weight_type: A string indicating which set of weights to use ('theoretical' or 'empirical').
        """
        if weight_type not in ['theoretical', 'empirical']:
            raise ValueError("weight_type must be either 'theoretical' or 'empirical'")
        
        self.active_weight_set = weight_type
        # Update the model's parameters based on the active weight set.
        if weight_type == 'theoretical':
            self.parameters.update(self.theoretical_weights)
        else:
            self.parameters.update(self.empirical_weights)

    def compute_saliency(self, image):
        """
        Computes the saliency map for a given image using the active weight set.
        Overrides the method from the BaseVisualAttention class to use the selected weight set.
        """
        # Before computing saliency, ensure the correct weights are set.
        self.set_active_weights(self.active_weight_set)

        # Proceed with computing the saliency map using the parent class's method.
        return super().compute_saliency(image)
    
    def run_theoretical_weights(self, image, disease_name):
        """
        Runs the model using the theoretical weights on a given image.
        """
        self.set_active_weights('theoretical')
        saliency_map = self.compute_saliency(image)
        self.save_results(saliency_map, disease_name, 'theoretical')

    def run_empirical_weights(self, image, disease_name):
        """
        Runs the model using the empirical weights on a given image.
        """
        self.set_active_weights('empirical')
        saliency_map = self.compute_saliency(image)
        self.save_results(saliency_map, disease_name, 'empirical')

    def run_full_experiment(self, image, disease_name):
        """
        Runs the full experiment using both sets of weights.
        """
        self.run_theoretical_weights(image, disease_name)
        self.run_empirical_weights(image, disease_name)

    def save_results(self, saliency_map, disease_name, weight_type):
        """
        Saves the results of the experiment in a specified format and as a grayscale photo.
        """
        # Ensure the results directory for the disease exists.
        results_dir = f'results/{disease_name}/{weight_type}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(results_dir, exist_ok=True)

        # Save the saliency map as a numpy file for later use.
        np.save(os.path.join(results_dir, 'saliency_map.npy'), saliency_map)

        # Convert the saliency map to a format suitable for saving as an image.
        normalized_saliency = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(results_dir, 'saliency_map.png'), normalized_saliency)
