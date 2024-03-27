import math
from base_model.VisualAttentionDefinitions import DEFAULT_PARAMETERS, IMAGE_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE
import cv2
import numpy as np
from scipy.ndimage import maximum_filter

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
        preprocessed_image = self.preprocess_image(image)
        self.intensity_features = self.extract_intensity_features(preprocessed_image)
        self.color_features = self.extract_color_features(preprocessed_image)
        self.orientation_features = self.extract_orientation_features(preprocessed_image)

        normalized_intensity = self.normalize_feature_maps(self.intensity_features)
        normalized_color = self.normalize_feature_maps(self.color_features)
        normalized_orientation = self.normalize_feature_maps(self.orientation_features)

        intensity_conspicuity = self.create_conspicuity_map(normalized_intensity)
        color_conspicuity = self.create_conspicuity_map(normalized_color)
        orientation_conspicuity = self.create_conspicuity_map(normalized_orientation)

        weights = self.parameters
        weighted_intensity = intensity_conspicuity * weights['intensity_weight']
        weighted_color = color_conspicuity * weights['color_weight']
        weighted_orientation = orientation_conspicuity * weights['orientation_weight']

        saliency_map = (weighted_intensity + weighted_color + weighted_orientation) / (weights['intensity_weight'] + weights['color_weight'] + weights['orientation_weight'])

        self.saliency_map = self.postprocess_saliency_map(saliency_map)
        self.saliency_map = cv2.resize(self.saliency_map, (image.shape[1], image.shape[0]))
        return self.saliency_map

    def preprocess_image(self, image, start_size=(1983, 1088)):
        preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_image = cv2.resize(preprocessed_image, start_size)
        preprocessed_image = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)
        return preprocessed_image


    def extract_intensity_features(self, image, levels=9):
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        pyramid = self.create_gaussian_pyramid(grayscale, levels)
        intensity_features = []
        for c in range(2, 5):
            center_surround_1 = self.center_surround_diff(pyramid, c, c + 3)
            center_surround_2 = self.center_surround_diff(pyramid, c, c + 4)
            intensity_features.append(center_surround_1)
            intensity_features.append(center_surround_2)
        return intensity_features

    def extract_color_features(self, image, levels=9):
        image = self.makeNormalizedColorChannels(image)
        r, g, b, y = cv2.split(image)
        r_pyramid = self.create_gaussian_pyramid(r, levels)
        g_pyramid = self.create_gaussian_pyramid(g, levels)
        b_pyramid = self.create_gaussian_pyramid(b, levels)
        y_pyramid = self.create_gaussian_pyramid(y, levels)
        rg_features = []
        by_features = []
        for c in range(2, 5):
            r_center_surround_1 = self.center_surround_diff(r_pyramid, c, c + 3)
            g_center_surround_1 = self.center_surround_diff(g_pyramid, c, c + 3)
            rg_features.append(cv2.absdiff(r_center_surround_1, g_center_surround_1))

            b_center_surround_1 = self.center_surround_diff(b_pyramid, c, c + 3)
            y_center_surround_1 = self.center_surround_diff(y_pyramid, c, c + 3)
            by_features.append(cv2.absdiff(b_center_surround_1, y_center_surround_1))

            r_center_surround_2 = self.center_surround_diff(r_pyramid, c, c + 4)
            g_center_surround_2 = self.center_surround_diff(g_pyramid, c, c + 4)
            rg_features.append(cv2.absdiff(r_center_surround_2, g_center_surround_2))

            b_center_surround_2 = self.center_surround_diff(b_pyramid, c, c + 4)
            y_center_surround_2 = self.center_surround_diff(y_pyramid, c, c + 4)
            by_features.append(cv2.absdiff(b_center_surround_2, y_center_surround_2))
        return rg_features + by_features

    def match_shapes_and_absdiff(self, arr1, arr2):
        if arr1.shape != arr2.shape:
            height, width = max(arr1.shape, arr2.shape)
            arr1 = cv2.resize(arr1, (width, height))
            arr2 = cv2.resize(arr2, (width, height))
        return cv2.absdiff(arr1, arr2)

    def extract_orientation_features(self, image, levels=9, num_orientations=4):
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        pyramid = self.create_gaussian_pyramid(grayscale, levels)
        orientation_features = []
        for c in range(2, 5):
            center = pyramid[c]
            for angle in range(0, 180, 180 // num_orientations):
                theta = angle * (math.pi / 180)
                gaborFilter = self.makeGaborFilter(dims=(10, 10), lambd=2.5, theta=theta, psi=math.pi/2, sigma=2.5, gamma=0.5)
                filtered = gaborFilter(center)
                orientation_features.append(filtered)
        return orientation_features
    
    def makeGaborFilter(self, dims, lambd, theta, psi, sigma, gamma):
        def xpf(i, j):
            return i * math.cos(theta) + j * math.sin(theta)
        def ypf(i, j):
            return -i * math.sin(theta) + j * math.cos(theta)
        def gabor(i, j):
            xp = xpf(i, j)
            yp = ypf(i, j)
            return math.exp(-(xp**2 + gamma**2 * yp**2) / (2 * sigma**2)) * math.cos(2 * math.pi * xp / lambd + psi)
        
        halfwidth = dims[0] // 2
        halfheight = dims[1] // 2

        kernel = np.array([[gabor(halfwidth - i, halfheight - j) for j in range(dims[1])] for i in range(dims[0])])

        def theFilter(image):
            return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

        return theFilter
    
    def makeNormalizedColorChannels(self, image, thresholdRatio=10.):
        intens = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = intens.max() / thresholdRatio
        r, g, b = cv2.split(image)
        cv2.threshold(src=r, dst=r, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=g, dst=g, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=b, dst=b, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
        R = r - (g + b) / 2
        G = g - (r + b) / 2
        B = b - (g + r) / 2
        Y = (r + g) / 2 - cv2.absdiff(r, g) / 2 - b

        cv2.threshold(src=R, dst=R, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=G, dst=G, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=B, dst=B, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
        cv2.threshold(src=Y, dst=Y, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)

        image = cv2.merge((R, G, B, Y))
        return image

    def create_conspicuity_map(self, feature_maps):
        conspicuity_map = np.zeros_like(feature_maps[0], dtype=np.float32)
        for fmap in feature_maps:
            resized_fmap = cv2.resize(fmap, (conspicuity_map.shape[1], conspicuity_map.shape[0]))
            conspicuity_map += resized_fmap
        conspicuity_map = cv2.normalize(conspicuity_map, None, 0, 1, cv2.NORM_MINMAX)
        return conspicuity_map

    def combine_conspicuity_maps(self, intensity_map, color_map, orientation_map):
        saliency_map = (intensity_map + color_map + orientation_map) / 3
        saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
        return saliency_map

    def postprocess_saliency_map(self, saliency_map, num_iters=10):
        for _ in range(num_iters):
            winner = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
            self.apply_inhibition_of_return(saliency_map, winner)
        
        normalized_saliency = cv2.normalize(saliency_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        kernel_size = self.parameters.get('smoothing_kernel_size', 5)
        smoothed_saliency = cv2.GaussianBlur(normalized_saliency, (kernel_size, kernel_size), 0)
        return smoothed_saliency
    
    def create_gaussian_pyramid(self, image, levels):
        pyramid = [image]
        for _ in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    def center_surround_diff(self, pyramid, center_level, surround_level):
        center = pyramid[center_level]
        surround = cv2.pyrUp(pyramid[surround_level])
        
        if len(center.shape) == 1:
            # 1-dimensional array, reshape to 2-dimensional array
            center = center.reshape((1, -1))
            surround = surround.reshape((1, -1))
        
        if len(center.shape) == 2:
            # Single-channel image (grayscale)
            height, width = center.shape
        elif len(center.shape) == 3:
            # Multi-channel image (color)
            height, width, _ = center.shape
        else:
            raise ValueError(f"Unexpected image shape: {center.shape}")
    
        surround = cv2.resize(surround, (width, height))
        diff = cv2.absdiff(center, surround)
        return diff
    
    def normalize_feature_maps(self, feature_maps):
        normalized_maps = []
        for fmap in feature_maps:
            normalized_map = self.N(fmap)
            normalized_maps.append(normalized_map)
        return normalized_maps
    
    def N(self, image):
        M = 8.  # an arbitrary global maximum to which the image is scaled.
        image = cv2.convertScaleAbs(image, alpha=M/image.max(), beta=0.)
        w, h = image.shape
        maxima = maximum_filter(image, size=(w//10, h//1))
        maxima = (image == maxima)
        mnum = maxima.sum()
        maxima = np.multiply(maxima, image)
        mbar = float(maxima.sum()) / mnum
        return image * (M - mbar)**2

    def find_local_maxima(self, feature_map, local_max_size):
        local_max_list = []
        for fm in feature_map:
            fm = np.asarray(fm, dtype=np.float32)
            fm = (fm * 255).astype(np.uint8)
            local_max = cv2.dilate(fm, np.ones((local_max_size, local_max_size), np.uint8))
            local_max = cv2.compare(fm, local_max, cv2.CMP_GE)
            local_max_list.append(local_max)
        return local_max_list

    def normalize_map(self, feature_map, local_max_list):
        fmap_max = np.max(feature_map)
        fmap_avg = np.mean(feature_map)
        
        # Iterate over local_max_list and update fmap_avg
        for local_max in local_max_list:
            if local_max.shape == feature_map.shape:
                fmap_avg = np.mean(feature_map[local_max != 0])
                break
            elif local_max.shape[:2] == feature_map.shape[:2]:
                fmap_avg = np.mean(feature_map[local_max[:, :, 0] != 0])
                break
        
        normalized_map = (feature_map - fmap_avg) / (fmap_max - fmap_avg + 1e-8)
        normalized_map = np.clip(normalized_map, 0, 1)
        return normalized_map
    
    def apply_inhibition_of_return(self, saliency_map, winner_location, inhibition_radius=50, inhibition_strength=0.2):
        height, width = saliency_map.shape
        y, x = winner_location

        # Create a circular mask for inhibition of return
        y_grid, x_grid = np.ogrid[-y:height-y, -x:width-x]
        mask = x_grid**2 + y_grid**2 <= inhibition_radius**2

        # Apply inhibition of return to the saliency map
        saliency_map[mask] *= (1 - inhibition_strength)