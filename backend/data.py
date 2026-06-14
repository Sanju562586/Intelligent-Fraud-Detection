import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath="creditcard.csv"):
    df = pd.read_csv(filepath)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_simulation_data(X_test, y_test):
    test_df = X_test.copy()
    test_df["Class"] = y_test
    normal_df = test_df[test_df["Class"] == 0]
    fraud_df = test_df[test_df["Class"] == 1]
    return normal_df, fraud_df
