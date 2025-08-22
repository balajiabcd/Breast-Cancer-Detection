import pytest
from src.data_utils import load_data, preprocess_data
import pandas as pd

DATA_PATH = "data/breast-cancer.csv"

def test_load_data():
    df = load_data(DATA_PATH)
    # Check dataframe is not empty
    assert len(df) > 0
    # Check required columns exist
    required_columns = ['diagnosis'] + [col for col in df.columns if col != 'diagnosis']
    for col in required_columns:
        assert col in df.columns
    # Check diagnosis is encoded as 0 or 1
    assert set(df['diagnosis'].unique()).issubset({0,1})

def test_preprocess_data():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Check no missing values
    import numpy as np
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    # Check target classes
    assert set(y_train.unique()).issubset({0,1})
    assert set(y_test.unique()).issubset({0,1})
