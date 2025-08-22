import unittest
from src.data_utils import load_data, preprocess_data
from src.model import train_and_evaluate
from sklearn.linear_model import LogisticRegression

class TestModel(unittest.TestCase):
    def test_pipeline_runs(self):
        df = load_data("data/breast-cancer.csv)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = LogisticRegression(max_iter=1000)
        train_and_evaluate(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    unittest.main()
