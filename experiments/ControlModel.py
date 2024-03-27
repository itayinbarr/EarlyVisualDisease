import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from base_model.DiseaseModel import DiseaseModel

class ControlModel(DiseaseModel):
    def __init__(self):
        # Initialize the base model with default parameters by not passing any specific weights.
        super().__init__(None, None)

def run_control_experiment(image_path, output_dir='results/Control'):
    """
    Runs the control experiment on the provided image using default model parameters.
    
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

    # Initialize the Control model.
    control_model = ControlModel()

    # Compute the saliency map using the default model parameters.
    saliency_map = control_model.compute_saliency(image)
    output_path = os.path.join(output_dir, 'default_saliency.png')
    cv2.imwrite(output_path, saliency_map * 255)  # Convert to 8-bit image format

    print(f"Control experiment completed successfully. Results saved to {output_dir}")

# If this script is run directly, ask for an image path and output directory, then run the experiment.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ControlExperiment.py <image_path> [<output_dir>]")
    else:
        img_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else 'results/Control'
        run_control_experiment(img_path, out_dir)
