from base_model.VisualAttentionDefinitions import DEFAULT_PARAMETERS, IMAGE_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE
import cv2
import numpy as np

class BaseVisualAttention:
    def __init__(self, parameters=None):
        """
        Initializes the base visual attention model with default or provided parameters.
        """
        # Load default parameters from VisualAttentionDefinitions.py or merge with provided ones.
        self.parameters = DEFAULT_PARAMETERS.copy()
        if parameters:
            self.parameters.update(parameters)

        # Initialize additional attributes needed for the model.
        self.image = None
        self.saliency_map = None
        self.intensity_features = None
        self.color_features = None
        self.orientation_features = None
        self.feature_extractors_initialized = False

    def compute_saliency(self, image):
        """
        Computes the saliency map for a given image.
        """
        self.image = image
        preprocessed_image = self.preprocess_image(image)
        self.intensity_features = self.extract_intensity_features(preprocessed_image)
        self.color_features = self.extract_color_features(preprocessed_image)
        self.orientation_features = self.extract_orientation_features(preprocessed_image)
        combined_saliency = self.combine_feature_maps(self.intensity_features, self.color_features, self.orientation_features)
        self.saliency_map = self.postprocess_saliency_map(combined_saliency)
        return self.saliency_map

    def preprocess_image(self, image):
        """
        Applies preprocessing steps to the input image.
        """
        preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_image = cv2.resize(preprocessed_image, IMAGE_SIZE)
        preprocessed_image = cv2.GaussianBlur(preprocessed_image, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        return preprocessed_image

    def extract_intensity_features(self, image):
        """
        Extracts intensity features from the preprocessed image.
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return grayscale

    def extract_color_features(self, image):
        """
        Extracts color features from the preprocessed image.
        """
        R, G, B = cv2.split(image)
        return R, G, B

    def extract_orientation_features(self, image):
        """
        Extracts orientation features from the preprocessed image.
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=5)
        orientation_features = cv2.magnitude(sobelx, sobely)
        return orientation_features

    def combine_feature_maps(self, intensity_features, color_features, orientation_features):
        """
        Combines extracted features into a single saliency map.
        """
        combined_features = np.mean(np.array([intensity_features, *color_features, orientation_features]), axis=0)
        return combined_features

    def postprocess_saliency_map(self, saliency_map):
        """
        Applies post-processing steps to the combined saliency map.
        """
        normalized_saliency = cv2.normalize(saliency_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Provide a default value for smoothing_kernel_size if it's not specified
        kernel_size = self.parameters.get('smoothing_kernel_size', 5)  # Use 5 as a default value
        smoothed_saliency = cv2.GaussianBlur(normalized_saliency, (kernel_size, kernel_size), 0)
        return smoothed_saliency

