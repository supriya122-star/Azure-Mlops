import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_data, preprocess_data

# MLflow Local Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Insurance-Premium-Prediction")

DATA_PATH = "data/raw/insurance.csv"

df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = preprocess_data(df)

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE : {mae}")
    print(f"R2  : {r2}")

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="insurance-model"
    )

    print("Model saved successfully.")