# Serve predictions with mlflow.sklearn.load_model()

import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from tabulate import tabulate
import mlflow

print("MLflow Version:", mlflow.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())


def read_prediction_data(data_path):
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_json(data_path)
    if "quality" in df:
        df = df.drop(["quality"], axis=1)
    print("Data:")
    print("  shape:", df.shape)
    print("  dtypes:")
    for x in zip(df.columns, df.dtypes):
        print(f"    {x[0]}: {x[1]}")
    return df


def display_predictions(predictions):
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    if isinstance(predictions, np.ndarray):
        predictions = np.round(predictions, decimals=3)
        predictions = pd.DataFrame(predictions, columns=["prediction"])
    else:
        predictions = predictions.round(3)
    df = predictions.head(5)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI DATA_PATH")
        sys.exit(1)
    model_uri = sys.argv[1]  # mlruns\1\61d45b785b0742168e609bf5029e1479\artifacts\sklearn-model
    data_path = sys.argv[2] if len(sys.argv) > 2 else "wine-quality-white.csv"
    print("data_path:", data_path)
    print("model_uri:", model_uri)

    model = mlflow.sklearn.load_model(model_uri)
    print("model.type:", type(model))

    data = read_prediction_data(data_path)
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)
