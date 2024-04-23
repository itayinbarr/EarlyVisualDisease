# Early Visual Disease Modeling and Itti-Koch Saliency Model Implementation

This project serves two main purposes:

1. Modeling early visual diseases by manipulating the weights of intensity, orientation, and color features in a saliency-based visual attention model.
2. Providing a well-documented and clearly written implementation of the Itti-Koch saliency model in Python.

The model is based on the Itti-Koch model, which is modeling the neuronal architecture of the early primate visual system. While there are existing bad or old implementations of the Itti-Koch model available in Python, this project aims to provide a clean, readable, and well-structured implementation, making it easier for researchers and developers to understand and build upon.

## Background

Visual attention plays a crucial role in the interpretation of complex scenes by selecting a subset of the available sensory information before further processing. The Itti-Koch model, proposed by Laurent Itti and Christof Koch, is a biologically-plausible computational model of visual attention that draws inspiration from the early visual pathways in the primate brain.

The model consists of several key components that mimic the processing stages in the early visual system:

1. **Feature Extraction**: The input image is decomposed into a set of topographic feature maps, including intensity, color, and orientation. These features are extracted using linear center-surround operations akin to visual receptive fields found in the retina, lateral geniculate nucleus (LGN), and primary visual cortex (V1).

   - Intensity features capture local contrast differences in the image, similar to the processing of achromatic information in the magnocellular pathway of the LGN and V1.
   - Color features are extracted using color-opponent mechanisms, resembling the processing of chromatic information in the parvocellular pathway of the LGN and V1. The model considers red-green and blue-yellow color opponency, which are prevalent in the early visual system.
   - Orientation features are extracted using Gabor filters at different scales and orientations, mimicking the processing of orientation-selective neurons in V1. These neurons are organized into orientation columns, each responding to a specific range of orientations.

2. **Conspicuity Maps**: The feature maps are then combined into three conspicuity maps: intensity, color, and orientation. These maps highlight the most salient regions in each feature dimension, similar to the processing in higher-order visual areas such as V2 and V4, where neurons have larger receptive fields and integrate information from multiple feature dimensions.

3. **Saliency Map**: The conspicuity maps are integrated into a single topographical saliency map, which represents the conspicuity or saliency at every location in the visual field. This integration process is thought to occur in the parietal and frontal cortical areas, where top-down attentional control signals interact with bottom-up saliency information.

4. **Attention Guidance**: The saliency map guides the selection of attended locations in the visual field through a winner-take-all (WTA) neural network. The WTA network is inspired by the competitive interactions among neurons in the visual cortex and the superior colliculus, where the most salient location is selected for further processing and attentional allocation.

The Itti-Koch model provides a biologically-plausible framework for understanding bottom-up visual attention and has been widely used in various applications. It captures the key aspects of early visual processing, such as the separation of visual information into different feature channels (intensity, color, and orientation), the hierarchical processing of features, and the integration of feature maps to guide attentional selection.

By incorporating these biologically-inspired components, the model offers insights into how the early visual pathways contribute to the emergence of visual saliency and how attention can be guided by the interplay between bottom-up and top-down processes.

## Itti-Koch Model Implementation

This project provides a clear and well-documented implementation of the Itti-Koch saliency model in Python. The implementation is designed to be modular, extensible, and easy to understand. It follows the original model architecture closely while leveraging modern Python libraries and best practices.
Key features of the implementation:

- Modular design with separate classes for feature extraction, conspicuity map creation, and saliency map computation.
- Well-documented code with clear explanations of each step in the model.
- Use of efficient NumPy and OpenCV operations for improved performance.
- Easily configurable parameters for customizing the model behavior.
- Compatibility with various input image sizes.

The implementation serves as a valuable resource for researchers and developers interested in understanding the Itti-Koch model and incorporating it into their own projects. It can be used as a starting point for further exploration, modifications, and extensions of the model.

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
- `results/`: Directory where the experiment results are stored. It is gitignored.
- `src/`: Directory containing the source code.
  - `analysis/`: Directory containing the analysis code.
    - `analysis.py`: Analysis class for calculating similarities and generating plots.
  - `base_model/`: Directory containing the base model code.
    - `BaseVisualAttention.py`: Base class for the visual attention model.
    - `DiseaseModel.py`: Disease-specific model class.
    - `VisualAttentionDefinitions.py`: Definitions and parameters for the visual attention model.
  - `tests/`: Directory containing test scripts.
  - `analysis_launcher.py`: Script to launch the analysis.
  - `experiments_launcher.py`: Script to run the experiments for all diseases on all images.
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

![Alt text](/flowmap.png?raw=true "Logic Flow Chart")

To run the experiments for all diseases on all images, use the `experiments_launcher.py` script:

```
python experiments_launcher.py
```

Alternatively, you can pick specific images and results directories of your choosing from the terminal:

```
python experiments_launcher.py <image_directory> <results_directory>
```

- `<image_directory>`: Path to the directory containing the input images.
- `<results_directory>`: Path to the directory where the experiment results will be stored.

### Adding New Diseases

To add a new disease model, follow these steps:

1. Create a new disease model file in the `experiments/` directory, e.g., `NewDisease.py`.
2. Define the theoretical and empirical weights for the disease in the model file.
3. Create a new disease model class extending the `DiseaseModel` class.
4. Implement the disease-specific experiment function in the model file.
5. Import the disease model in `experiments_launcher.py` and add it to the experiments list.

### Running Analysis

To run the analysis and generate plots, use the `analysis_launcher.py` script:

```
python analysis_launcher.py
```

The script will generate plots for disease output similarity and weight distance between empirical and theoretical weights.

### Using the Itti-Koch Model Implementation

The Itti-Koch model implementation can be found in the src/base_model/ directory. To use the model in your own projects:

1. Import the necessary classes from the base_model module.
2. Create an instance of the BaseVisualAttention class.
3. Preprocess your input image using the preprocess_image method.
4. Extract features using the extract_intensity_features, extract_color_features, and extract_orientation_features methods.
5. Normalize the feature maps using the normalize_feature_maps method.
6. Create conspicuity maps using the create_conspicuity_map method.
7. Combine the conspicuity maps to obtain the saliency map.
8. Postprocess the saliency map using the postprocess_saliency_map method.

Refer to the documentation in the BaseVisualAttention class for more details on each method and its parameters.

## References

- Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254-1259.

---
