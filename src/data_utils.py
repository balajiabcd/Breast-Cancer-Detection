import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    df = pd.read_csv(path)
    df.diagnosis = df.diagnosis.map({"M": 1, "B": 0})
    return df

def preprocess_data(df):
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

