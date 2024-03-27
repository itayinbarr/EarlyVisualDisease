import os
import cv2
from DiseaseModel import DiseaseModel

# Define the theoretical and empirical weights for Macular Degeneration.
THEORETICAL_WEIGHTS = {
    'intensity_weight': 0.8,  # Hypothetical value, adjust based on research
    'color_weight': 0.5,      # Hypothetical value, adjust based on research
    'orientation_weight': 0.7, # Hypothetical value, adjust based on research
}

EMPIRICAL_WEIGHTS = {
    'intensity_weight': 0.75, # Hypothetical value, adjust based on experimental findings
    'color_weight': 0.55,     # Hypothetical value, adjust based on experimental findings
    'orientation_weight': 0.65, # Hypothetical value, adjust based on experimental findings
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
    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Initialize the Macular Degeneration model.
    md_model = MacularDegenerationModel()

    # Run the experiment with theoretical weights.
    md_model.set_active_weights('theoretical')
    theoretical_saliency = md_model.compute_saliency(image)
    theoretical_output_path = os.path.join(output_dir, 'theoretical_saliency.png')
    cv2.imwrite(theoretical_output_path, theoretical_saliency * 255)  # Convert to 8-bit image format

    # Run the experiment with empirical weights.
    md_model.set_active_weights('empirical')
    empirical_saliency = md_model.compute_saliency(image)
    empirical_output_path = os.path.join(output_dir, 'empirical_saliency.png')
    cv2.imwrite(empirical_output_path, empirical_saliency * 255)  # Convert to 8-bit image format

    print(f"Experiment completed successfully. Results saved to {output_dir}")

# If this script is run directly, ask for an image path and output directory, then run the experiment.
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python MacularDegeneration.py <image_path> [<output_dir>]")
    else:
        img_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/MacularDegeneration'
        run_macular_degeneration_experiment(img_path, out_dir)
