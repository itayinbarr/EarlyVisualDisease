"""
experiments_launcher.py

This module provides functionality to run experiments for all diseases on all images in a specified directory.
It imports the disease models and defines a function to run all experiments. The module also sets up logging
to track the progress of the experiments.
"""
import os
import glob
import sys
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import disease models here
from experiments.Cataracts import run_cataracts_experiment
from experiments.ControlModel import run_control_experiment
from experiments.MacularDegeneration import run_macular_degeneration_experiment
from experiments.DiabeticRetinopathy import run_diabeticretinopathy_experiment
from experiments.Glaucoma import run_glaucoma_experiment
from experiments.OpticNeuritis import run_opticneuritis_experiment
from experiments.RetinitisPigmentosa import run_retinitispigmentosa_experiment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_experiments(image_directory, results_directory):
    """
    Runs experiments for all diseases on all images in the specified directory.

    Args:
        image_directory (str): The directory containing the images to process.
        results_directory (str): The root directory where results will be stored.
    """
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
        logger.info(f"Created results directory: {results_directory}")
    else:
        logger.info(f"Results directory already exists: {results_directory}")

    image_paths = glob.glob(os.path.join(image_directory, '*.png'))  
    logger.info(f"Found {len(image_paths)} images in directory: {image_directory}")

    # Define the experiments to run.
    # Each experiment should be a tuple consisting of (experiment_function, disease_name).
    experiments = [
        (run_control_experiment, 'ControlModel'),
        (run_macular_degeneration_experiment, 'MacularDegeneration'),
        (run_cataracts_experiment, 'Cataract'),
        (run_diabeticretinopathy_experiment, 'DiabeticRetinopathy'),
        (run_glaucoma_experiment, 'Glaucoma'),
        (run_opticneuritis_experiment, 'OpticNeuritis'),
        (run_retinitispigmentosa_experiment, 'RetinitisPigmentosa')
    ]
    logger.info(f"Defined {len(experiments)} experiments to run")

    for image_path in image_paths:
        logger.info(f"Processing image: {image_path}")
        for experiment_function, disease_name in experiments:
            logger.info(f"Running {disease_name} experiment on {image_path}")
            output_dir = os.path.join(results_directory, disease_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            logger.info(f"Created output directory for experiment: {output_dir}")
            experiment_function(image_path, output_dir)
            logger.info(f"Completed {disease_name} experiment on {image_path}")
        logger.info(f"Finished processing image: {image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        img_dir = './data/'
        res_dir = './results'
    else:
        img_dir = sys.argv[1]
        res_dir = sys.argv[2]

    logger.info(f"Starting experiments with image directory: {img_dir} and results directory: {res_dir}")
    run_all_experiments(img_dir, res_dir)
    logger.info("All experiments completed")