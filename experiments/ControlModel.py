"""
ControlModel.py

This module provides the ControlModel class, which extends the DiseaseModel class to serve as a control model
for visual attention experiments. It defines the default theoretical and empirical weights for the control model
and provides a function to run the control experiment on a given image.
"""
import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base_model.DiseaseModel import DiseaseModel

# Define the theoretical and empirical weights for the Control Model.
THEORETICAL_WEIGHTS = {
    'intensity_weight': 1.0,  # Default value for the control model
    'color_weight': 1.0,      # Default value for the control model
    'orientation_weight': 1.0,  # Default value for the control model
}

EMPIRICAL_WEIGHTS = {
    'intensity_weight': 1.0,  # Default value for the control model
    'color_weight': 1.0,      # Default value for the control model
    'orientation_weight': 1.0,  # Default value for the control model
}

class ControlModel(DiseaseModel):
    """
    The ControlModel class extends the DiseaseModel class to serve as a control model for visual attention experiments.
    It uses the default theoretical and empirical weights for the control model.
    """
    def __init__(self):
        """
        Initializes the ControlModel with the default theoretical and empirical weights for the control model.
        """
        super().__init__(THEORETICAL_WEIGHTS, EMPIRICAL_WEIGHTS)

def run_control_experiment(image_path, output_dir='results/Control'):
    """
    Runs the control experiment on the provided image using default model parameters.

    Args:
        image_path (str): Path to the image file to process.
        output_dir (str, optional): Directory where the experiment results will be saved.
            Defaults to 'results/Control'.
    """
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract the image name.
    image_name = os.path.basename(image_path)

    control_model = ControlModel()
    control_model.run_full_experiment(image, 'Control', image_name)

