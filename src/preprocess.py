import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    X = df.drop("charges", axis=1)
    y = df["charges"]

    categorical_cols = ["sex", "smoker", "region"]

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test