# Visual Attention Disease Model Experiments Repository

## Overview

This repository contains an implementation of a visual attention model based on the Itti-Koch framework. The core functionality revolves around simulating how human vision highlights areas of interest within images, particularly focusing on aspects like intensity, color, and orientation features. The repository is structured to facilitate running experiments with models like ControlModel and MacularDegeneration, among others, using real-world image data.

## Getting Started

To get started with the repository:

1. **Clone the Repository**: Clone this repository to your local machine using your preferred method (HTTPS, SSH, GitHub CLI).

2. **Environment Setup**: Ensure you have Python installed. While the code is primarily based on Python 3, pay attention to specific dependencies such as OpenCV, NumPy, and SciPy. Install these via pip:

   ```bash
   pip install numpy opencv-python scipy
   ```

3. **Prepare Your Data**: Place your image datasets within the `./data/` directory. The current setup assumes images are in PNG format. Adjust if necessary to match your data format.

## Running Experiments

The `ExperimentRunner.py` script is the main entry point for running experiments:

1. **Basic Usage**: To run the experiments with default settings (using images from `./data/` and saving results in `./results/`), simply execute the script without any arguments:

   ```bash
   python ExperimentRunner.py
   ```

2. **Custom Directories**: To specify custom directories for input images and results:
   ```bash
   python ExperimentRunner.py path/to/image_directory path/to/results_directory
   ```

### What Happens in an Experiment?

Each experiment processes images to generate saliency maps based on different disease models or control conditions:

- **Macular Degeneration Experiment**: Analyzes how macular degeneration might affect visual attention on images.
- **Control Experiment**: (Commented out by default) Can be enabled to compare with a standard or healthy visual attention model.

Results for each image and model will be stored in separate directories within the specified results directory, organized by the experiment type and timestamp.

## Repository Structure

- `experiments/`: Contains Python modules for each of the disease models.
- `data/`: Default directory for storing input images for experiments.
- `results/`: Default directory where experiment outputs are saved.

## Modifying Experiments

You can add or modify experiments by editing the `ExperimentRunner.py` script. Specifically, you can add new models in the `experiments` directory and include them in the `experiments` list within `run_all_experiments` function.

## Logging

The script provides detailed logging for each step of the process. Check the console output to monitor the progress and troubleshoot if necessary.

---

By using this repository, you can explore different aspects of visual attention and how various conditions may alter perception within visual scenes.
