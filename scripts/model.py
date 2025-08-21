from data_utils import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    df = load_data("scripts\breast-cancer.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate(rf, X_train, X_test, y_train, y_test)
