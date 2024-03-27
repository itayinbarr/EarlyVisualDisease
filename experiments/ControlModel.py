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
    def __init__(self):
        super().__init__(THEORETICAL_WEIGHTS, EMPIRICAL_WEIGHTS)

def run_control_experiment(image_path, output_dir='results/Control'):
    """
    Runs the control experiment on the provided image using default model parameters.
    
    :param image_path: Path to the image file to process.
    :param output_dir: Directory where the experiment results will be saved.
    """
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract the image name.
    image_name = os.path.basename(image_path)

    # Initialize the Control model and run the experiment.
    control_model = ControlModel()
    control_model.run_full_experiment(image, 'Control', image_name)

# If this script is run directly, ask for an image path and output directory, then run the experiment.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ControlModel.py <image_path> [<output_dir>]")
    else:
        img_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/Control'
        run_control_experiment(img_path, out_dir)