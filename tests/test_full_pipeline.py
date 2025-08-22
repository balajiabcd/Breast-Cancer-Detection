import pytest
import os
from src.data_utils import load_data, preprocess_data
from src.model import train_and_evaluate
from src.visualize import plot_correlation, plot_class_distribution
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/breast-cancer.csv"
MODELS_DIR = "models"

# Ensure models folder exists for saving
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Ensure images folder exists
IMAGES_DIR = "images"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

def test_full_pipeline():
    # 1. Load data
    df = load_data(DATA_PATH)
    assert not df.empty, "Data should not be empty"

    # 2. Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert X_train.shape[0] == y_train.shape[0], "Train features and target length mismatch"
    assert X_test.shape[0] == y_test.shape[0], "Test features and target length mismatch"

    # 3. Train and evaluate model
    model_name = "pipeline_rf_test"
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    train_and_evaluate(model_name, rf, X_train, X_test, y_train, y_test)

    # 4. Check if model file is saved
    model_path = os.path.join(MODELS_DIR, model_name + ".pkl")
    assert os.path.exists(model_path), "Model file was not saved"

    # 5. Run visualizations (ensure no errors)
    plot_class_distribution(df)
    plot_correlation(df)
