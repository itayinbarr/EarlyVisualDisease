"""
analysis_launcher.py

This module initializes the Analysis class and uses it to perform the analysis of the experiment results.
It loads the experiment results from the specified directory and generates the desired plots.
"""

import os
import sys

from analysis.analysis import Analysis
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))



def main():
    results_directory = './results/'

    analysis = Analysis(results_directory)

    # Generate the plot for disease output similarity
    # analysis.plot_disease_similarity()

    # Generate the plot for weight distance (example for Macular Degeneration)
    analysis.plot_weight_distance('MacularDegeneration')

if __name__ == '__main__':
    main()