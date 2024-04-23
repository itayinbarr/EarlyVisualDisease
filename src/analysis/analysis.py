"""
analysis.py

This module provides the Analysis class, which handles the analysis of the experiment results.
It includes methods for calculating the similarity between disease outputs and generating plots
to visualize the results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    """
    The Analysis class handles the analysis of the experiment results.
    It provides methods for calculating the similarity between disease outputs and generating plots.
    """
    def __init__(self, results_directory):
        """
        Initializes the Analysis class with the results directory.
        
        Args:
            results_directory (str): The directory containing the experiment results.
        """
        self.results_directory = results_directory

    def get_latest_result(self, disease, image_name, weight_type):
        """
        Retrieves the path to the latest saliency map for a given disease, image, and weight type.
        
        Args:
            disease (str): The name of the disease.
            image_name (str): The name of the image.
            weight_type (str): The type of weights ('empirical' or 'theoretical').
            
        Returns:
            str: The path to the latest saliency map.
        """
        disease_folder = os.path.join(self.results_directory, disease, weight_type, image_name)
        timestamp_folders = os.listdir(disease_folder)
        latest_timestamp = max(timestamp_folders)
        saliency_map_path = os.path.join(disease_folder, latest_timestamp, 'saliency_map.npy')
        return saliency_map_path

    def calculate_similarity(self, disease1, disease2, image_name):
        """
        Calculates the similarity between the outputs of two diseases for a given image.
        
        Args:
            disease1 (str): The name of the first disease.
            disease2 (str): The name of the second disease.
            image_name (str): The name of the image.
            
        Returns:
            float: The similarity score between the disease outputs.
        """
        # Load the latest saliency maps for the two diseases
        saliency_map1_path = self.get_latest_result(disease1, image_name, 'empirical')
        saliency_map2_path = self.get_latest_result(disease2, image_name, 'empirical')
        
        saliency_map1 = np.load(saliency_map1_path)
        saliency_map2 = np.load(saliency_map2_path)
        
        # Calculate the similarity using a suitable metric (e.g., correlation)
        similarity = np.corrcoef(saliency_map1.flatten(), saliency_map2.flatten())[0, 1]
        
        return similarity

    def plot_disease_similarity(self):
        """
        Plots the similarity between disease outputs for each image.
        """
        # Get the list of images and diseases
        images = os.listdir(os.path.join(self.results_directory, next(os.walk(self.results_directory))[1][0], 'empirical'))
        diseases = next(os.walk(self.results_directory))[1]

        # Calculate the similarities for each image
        similarities = []
        for image in images:
            max_similarity = -1
            min_similarity = 1
            max_diseases = None
            min_diseases = None
            for i in range(len(diseases)):
                for j in range(i+1, len(diseases)):
                    similarity = self.calculate_similarity(diseases[i], diseases[j], image)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_diseases = (diseases[i], diseases[j])
                    if similarity < min_similarity:
                        min_similarity = similarity
                        min_diseases = (diseases[i], diseases[j])
            similarities.append((image, max_similarity, max_diseases, min_similarity, min_diseases))

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(images))
        bar_width = 0.35
        ax.bar(x - bar_width/2, [s[1] for s in similarities], bar_width, label='Most Similar')
        ax.bar(x + bar_width/2, [s[3] for s in similarities], bar_width, label='Least Similar')
        
        # Add disease labels to the bars
        for i, s in enumerate(similarities):
            ax.text(i - bar_width/2, s[1] + 0.01, f"{s[2][0]}\n{s[2][1]}", ha='center', va='bottom', fontsize=8, rotation=90)
            ax.text(i + bar_width/2, s[3] + 0.01, f"{s[4][0]}\n{s[4][1]}", ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in similarities])
        ax.set_ylabel('Similarity')
        ax.set_title('Disease Output Similarity')
        ax.legend()

        # Adjust the bottom margin to make room for the disease labels
        plt.margins(0.05, 0.3)

        plt.tight_layout()
        plt.show()

    def plot_weight_distance(self, disease):
        """
        Plots the distance between the empirical and theoretical weights for a given disease.
        
        Args:
            disease (str): The name of the disease.
        """
        # Load the latest empirical and theoretical saliency maps
        empirical_maps = []
        theoretical_maps = []
        image_names = []
        for image in os.listdir(os.path.join(self.results_directory, disease, 'empirical')):
            empirical_map_path = self.get_latest_result(disease, image, 'empirical')
            theoretical_map_path = self.get_latest_result(disease, image, 'theoretical')
            empirical_maps.append(np.load(empirical_map_path))
            theoretical_maps.append(np.load(theoretical_map_path))
            image_names.append(image)

        # Calculate the distance between the empirical and theoretical maps for each image
        distances = []
        for emp_map, theo_map in zip(empirical_maps, theoretical_maps):
            distance = np.linalg.norm(emp_map - theo_map)
            distances.append(distance)

        # Calculate the average distance
        avg_distance = np.mean(distances)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(image_names))
        ax.bar(x, distances)
        ax.axhline(avg_distance, color='red', linestyle='--', label='Average Distance')
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.set_ylabel('Distance')
        ax.set_title(f'Distance between Empirical and Theoretical Weights for {disease}')
        ax.legend()

        plt.tight_layout()
        plt.show()