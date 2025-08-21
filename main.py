from scripts.data_utils import load_data, preprocess_data
from scripts.model import train_and_evaluate
from scripts.visualize import plot_correlation, plot_class_distribution
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("scripts/breast-cancer.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Visualizations
    plot_class_distribution(df)
    plot_correlation(df)

    # Train and evaluate model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate(rf, X_train, X_test, y_train, y_test)
