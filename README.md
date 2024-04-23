Here's a comprehensive README.md file for your project:

# Early Visual Disease Modeling

This project aims to model early visual diseases by manipulating the weights of intensity, orientation, and color features in a saliency-based visual attention model. The model is inspired by the Itti-Koch model, which is based on the neuronal architecture of the early primate visual system.

## Background

Visual attention plays a crucial role in the interpretation of complex scenes by selecting a subset of the available sensory information before further processing. The Itti-Koch model, proposed by Laurent Itti and Christof Koch, is a biologically-plausible computational model of visual attention. It is inspired by the behavior and the neuronal architecture of the early primate visual system.

The model consists of several key components:

1. **Feature Extraction**: The input image is decomposed into a set of topographic feature maps, including intensity, color, and orientation. These features are extracted using linear center-surround operations akin to visual receptive fields.

2. **Conspicuity Maps**: The feature maps are then combined into three conspicuity maps: intensity, color, and orientation. These maps highlight the most salient regions in each feature dimension.

3. **Saliency Map**: The conspicuity maps are integrated into a single topographical saliency map, which represents the conspicuity or saliency at every location in the visual field.

4. **Attention Guidance**: The saliency map guides the selection of attended locations in the visual field through a winner-take-all neural network.

The Itti-Koch model provides a biologically-plausible framework for understanding bottom-up visual attention and has been widely used in various applications.

## Project Structure

The project is structured as follows:

- `data/`: Directory containing the input images for the experiments.
- `experiments/`: Directory containing the disease-specific model files.
  - `Cataracts.py`: Cataracts disease model.
  - `ControlModel.py`: Control model for comparison.
  - `DiabeticRetinopathy.py`: Diabetic Retinopathy disease model.
  - `Glaucoma.py`: Glaucoma disease model.
  - `MacularDegeneration.py`: Macular Degeneration disease model.
  - ...
- `results/`: Directory where the experiment results are stored.
- `src/`: Directory containing the source code.
  - `analysis/`: Directory containing the analysis code.
    - `analysis.py`: Analysis class for calculating similarities and generating plots.
    - `analysis_launcher.py`: Script to launch the analysis.
  - `base_model/`: Directory containing the base model code.
    - `BaseVisualAttention.py`: Base class for the visual attention model.
    - `DiseaseModel.py`: Disease-specific model class.
    - `VisualAttentionDefinitions.py`: Definitions and parameters for the visual attention model.
  - `tests/`: Directory containing test files.
- `ExperimentRunner.py`: Script to run the experiments for all diseases on all images.
- `README.md`: Project documentation.
- `REQUIREMENTS.txt`: List of required dependencies.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- SciPy

To install the dependencies, run:

```
pip install -r REQUIREMENTS.txt
```

## Usage

### Running Experiments

To run the experiments for all diseases on all images, use the `ExperimentRunner.py` script:

```
python ExperimentRunner.py <image_directory> <results_directory>
```

- `<image_directory>`: Path to the directory containing the input images.
- `<results_directory>`: Path to the directory where the experiment results will be stored.

### Adding New Diseases

To add a new disease model, follow these steps:

1. Create a new disease model file in the `experiments/` directory, e.g., `NewDisease.py`.
2. Define the theoretical and empirical weights for the disease in the model file.
3. Create a new disease model class extending the `DiseaseModel` class.
4. Implement the disease-specific experiment function in the model file.
5. Import the disease model in `ExperimentRunner.py` and add it to the experiments list.

### Running Analysis

To run the analysis and generate plots, use the `analysis_launcher.py` script:

```
python analysis_launcher.py
```

The script will generate plots for disease output similarity and weight distance between empirical and theoretical weights.

## Model Flow Chart

The following flow chart illustrates the logic of the visual attention model:

[Insert your flow chart image here]

## References

- Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254-1259.

---
