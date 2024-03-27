import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base_model.DiseaseModel import DiseaseModel

# Define the theoretical and empirical weights for Macular Degeneration.
THEORETICAL_WEIGHTS = {
    'intensity_weight': 1,  # Hypothetical value, adjust based on research
    'color_weight': 0.9,      # Hypothetical value, adjust based on research
    'orientation_weight': 0.9, # Hypothetical value, adjust based on research
}

EMPIRICAL_WEIGHTS = {
    'intensity_weight': 0.1, # Hypothetical value, adjust based on experimental findings
    'color_weight': 0.1,     # Hypothetical value, adjust based on experimental findings
    'orientation_weight': 0.4, # Hypothetical value, adjust based on experimental findings
}

class MacularDegenerationModel(DiseaseModel):
    def __init__(self):
        super().__init__(THEORETICAL_WEIGHTS, EMPIRICAL_WEIGHTS)

def run_macular_degeneration_experiment(image_path, output_dir='results/MacularDegeneration'):
    """
    Runs the Macular Degeneration experiment on the provided image.
    
    :param image_path: Path to the image file to process.
    :param output_dir: Directory where the experiment results will be saved.
    """
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract the image name.
    image_name = os.path.basename(image_path)

    # Initialize the Macular Degeneration model and run the experiment.
    md_model = MacularDegenerationModel()
    md_model.run_full_experiment(image, 'MacularDegeneration', image_name)

# If this script is run directly, ask for an image path and output directory, then run the experiment.
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python MacularDegeneration.py <image_path> [<output_dir>]")
    else:
        img_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/MacularDegeneration'
        run_macular_degeneration_experiment(img_path, out_dir)