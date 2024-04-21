"""
RetinitisPigmentosa.py

This module provides the RetinitisPigmentosaModel class, which extends the DiseaseModel class to model
visual attention for Retinitis Pigmentosa. It defines the theoretical and empirical weights specific to
Retinitis Pigmentosa and provides a function to run the Retinitis Pigmentosa experiment on a given image.
"""

import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from base_model.DiseaseModel import DiseaseModel

# Define the theoretical and empirical weights for Retinitis Pigmentosa.
THEORETICAL_WEIGHTS = {
    'intensity_weight': 0.5,  # Hypothetical value, adjust based on research
    'color_weight': 0.6,      # Hypothetical value, adjust based on research
    'orientation_weight': 0.8, # Hypothetical value, adjust based on research
}

EMPIRICAL_WEIGHTS = {
    'intensity_weight': 1, # Hypothetical value, adjust based on experimental findings
    'color_weight': 1,     # Hypothetical value, adjust based on experimental findings
    'orientation_weight': 1, # Hypothetical value, adjust based on experimental findings
}

class RetinitisPigmentosaModel(DiseaseModel):
    """
    The RetinitisPigmentosaModel class extends the DiseaseModel class to model visual attention for Retinitis Pigmentosa.
    It uses the theoretical and empirical weights specific to Retinitis Pigmentosa.
    """
    def __init__(self):
        """
        Initializes the RetinitisPigmentosaModel with the theoretical and empirical weights for Retinitis Pigmentosa.
        """
        super().__init__(THEORETICAL_WEIGHTS, EMPIRICAL_WEIGHTS)

def run_retinitispigmentosa_experiment(image_path, output_dir='results/RetinitisPigmentosa'):
    """
    Runs the Retinitis Pigmentosa experiment on the provided image.

    Args:
        image_path (str): Path to the image file to process.
        output_dir (str, optional): Directory where the experiment results will be saved.
            Defaults to 'results/RetinitisPigmentosa'.
    """
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract the image name.
    image_name = os.path.basename(image_path)

    md_model = RetinitisPigmentosaModel()
    md_model.run_full_experiment(image, 'RetinitisPigmentosa', image_name)

