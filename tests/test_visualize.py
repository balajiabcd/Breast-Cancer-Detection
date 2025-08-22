import pytest
import os
from src.visualize import plot_correlation, plot_class_distribution
from src.data_utils import load_data

DATA_PATH = "data/breast-cancer.csv"
IMAGES_DIR = "images"

# Ensure images folder exists
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

def test_plot_correlation_runs():
    df = load_data(DATA_PATH)
    # Just run the function to ensure no error
    plot_correlation(df)

def test_plot_class_distribution_runs():
    df = load_data(DATA_PATH)
    # Just run the function to ensure no error
    plot_class_distribution(df)
