import pytest
from src.data_utils import load_data, preprocess_data
from src.model import train_and_evaluate
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = "data/breast-cancer.csv"
MODELS_DIR = "models"

# Ensure models folder exists for saving
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def test_train_and_evaluate_runs():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model_name = "test_rf"
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    # This should run without errors
    train_and_evaluate(model_name, rf, X_train, X_test, y_train, y_test)
    # Check if model file is created
    model_path = os.path.join(MODELS_DIR, model_name + ".pkl")
    assert os.path.exists(model_path)
    # Load the model and predict
    loaded_model = joblib.load(model_path)
    y_pred = loaded_model.predict(X_test)
    assert len(y_pred) == len(y_test)
