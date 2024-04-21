"""
OpticNeuritis.py

This module provides the OpticNeuritisModel class, which extends the DiseaseModel class to model
visual attention for Optic Neuritis. It defines the theoretical and empirical weights specific to
Optic Neuritis and provides a function to run the Optic Neuritis experiment on a given image.
"""

import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from base_model.DiseaseModel import DiseaseModel

# Define the theoretical and empirical weights for Optic Neuritis.
THEORETICAL_WEIGHTS = {
    'intensity_weight': 0.85,  # Hypothetical value, adjust based on research
    'color_weight': 0.7,      # Hypothetical value, adjust based on research
    'orientation_weight': 0.9, # Hypothetical value, adjust based on research
}

EMPIRICAL_WEIGHTS = {
    'intensity_weight': 1, # Hypothetical value, adjust based on experimental findings
    'color_weight': 1,     # Hypothetical value, adjust based on experimental findings
    'orientation_weight': 1, # Hypothetical value, adjust based on experimental findings
}

class OpticNeuritisModel(DiseaseModel):
    """
    The OpticNeuritisModel class extends the DiseaseModel class to model visual attention for Optic Neuritis.
    It uses the theoretical and empirical weights specific to Optic Neuritis.
    """
    def __init__(self):
        """
        Initializes the OpticNeuritisModel with the theoretical and empirical weights for Optic Neuritis.
        """
        super().__init__(THEORETICAL_WEIGHTS, EMPIRICAL_WEIGHTS)

def run_cataracts_experiment(image_path, output_dir='results/OpticNeuritis'):
    """
    Runs the Optic Neuritis experiment on the provided image.

    Args:
        image_path (str): Path to the image file to process.
        output_dir (str, optional): Directory where the experiment results will be saved.
            Defaults to 'results/OpticNeuritis'.
    """
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract the image name.
    image_name = os.path.basename(image_path)

    md_model = OpticNeuritisModel()
    md_model.run_full_experiment(image, 'OpticNeuritis', image_name)

