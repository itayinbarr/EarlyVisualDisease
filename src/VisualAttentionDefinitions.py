# VisualAttentionDefinitions.py

# Define default parameters for the BaseVisualAttention model.
# These parameters can include weights for different features like intensity, color, and orientation,
# as well as other model-specific settings such as scale levels and threshold values.

DEFAULT_PARAMETERS = {
    'intensity_weight': 1.0,
    'color_weight': 1.0,
    'orientation_weight': 1.0,
    'scale_levels': [1, 2, 4],  # Example scales at which to analyze the image.
    # Add other parameters as needed.
}

# Define any global constants used in the model. This could include color spaces,
# standard image sizes, filter sizes, or other constants that are used across different parts of the model.

# Example:
IMAGE_SIZE = (640, 480)  # Default image size (width, height).
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)  # Kernel size for Gaussian blur.
COLOR_SPACE = 'RGB'  # Default color space for input images.

# Feature-specific constants can also be defined here.
# For example, you might specify the orientations and scales used for Gabor filters
# in the orientation feature extraction:

GABOR_FILTER_ORIENTATIONS = [0, 45, 90, 135]  # Angles in degrees.
GABOR_FILTER_SCALES = [1, 2, 3]  # Scale levels.

# You can also define thresholds and other parameters specific to feature extraction
# and saliency map computation:

INTENSITY_THRESHOLD = 0.5  # Example threshold for intensity feature extraction.
COLOR_THRESHOLD = 0.5  # Example threshold for color feature extraction.
ORIENTATION_THRESHOLD = 0.5  # Example threshold for orientation feature extraction.

# If your model includes normalization or post-processing steps, parameters for these
# can also be included:

NORMALIZATION_FACTOR = 1.0  # Example normalization factor.
SMOOTHING_KERNEL_SIZE = (3, 3)  # Kernel size for smoothing operations.

# This structure allows you to easily modify the model's behavior by adjusting these values,
# and provides a clear overview of all the settings and constants that the model uses.
