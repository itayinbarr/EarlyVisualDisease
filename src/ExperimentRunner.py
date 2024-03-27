import os
import cv2
from BaseVisualAttention import BaseVisualAttention
from VisualAttentionDefinitions import DEFAULT_PARAMETERS
import numpy as np

class ExperimentRunner:
    def __init__(self, image_directory, results_directory, diseases_configs, base_parameters=DEFAULT_PARAMETERS):
        """
        Initializes the experiment runner with directories and settings.
        
        :param image_directory: Directory containing subdirectories for each disease, each with images to process.
        :param results_directory: Directory where experiment results will be saved.
        :param diseases_configs: A dictionary mapping disease names to their specific configurations.
        :param base_parameters: Base parameters for the visual attention model, used if not overridden by disease-specific configs.
        """
        self.image_directory = image_directory
        self.results_directory = results_directory
        self.diseases_configs = diseases_configs
        self.base_parameters = base_parameters

        # Ensure result directories exist.
        self._prepare_result_directories()

    def _prepare_result_directories(self):
        """
        Ensures that the directory structure for storing results is present.
        """
        for disease in self.diseases_configs:
            disease_dir = os.path.join(self.results_directory, disease)
            if not os.path.exists(disease_dir):
                os.makedirs(disease_dir)

    def run_experiments(self):
        """
        Executes the batch processing of images for each disease condition.
        """
        for disease, config in self.diseases_configs.items():
            print(f"Running experiments for {disease}...")
            # Merge base parameters with disease-specific parameters.
            parameters = {**self.base_parameters, **config.get('parameters', {})}
            visual_attention_model = BaseVisualAttention(parameters)
            
            # Process each image in the disease-specific directory.
            disease_image_dir = os.path.join(self.image_directory, disease)
            for image_name in os.listdir(disease_image_dir):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(disease_image_dir, image_name)
                    image = cv2.imread(image_path)
                    saliency_map = visual_attention_model.compute_saliency(image)
                    
                    # Save the resulting saliency map.
                    result_path = os.path.join(self.results_directory, disease, f'saliency_{image_name}')
                    cv2.imwrite(result_path, saliency_map * 255)  # Convert to 0-255 scale for saving.
            print(f"Completed experiments for {disease}.")

# Usage example (assuming this script is executed directly):
if __name__ == '__main__':
    IMAGE_DIR = 'path/to/images'
    RESULTS_DIR = 'path/to/results'
    DISEASES_CONFIGS = {
        'Disease1': {'parameters': {'intensity_weight': 1.2, 'color_weight': 0.8}},
        'Disease2': {'parameters': {'intensity_weight': 1.0, 'color_weight': 1.0}},
        # Add more diseases and their configurations here.
    }

    runner = ExperimentRunner(IMAGE_DIR, RESULTS_DIR, DISEASES_CONFIGS)
    runner.run_experiments()
