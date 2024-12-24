import threading
from flask import jsonify
import pandas as pd
import numpy as np
from app.services.model_service import save_model_metadata, update_model_status
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import os

MODEL_DIR = os.path.join(os.getcwd(), "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model_thread(df, model_name):
    try:
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        if df.empty:
            raise ValueError("Dataset is empty after preprocessing. Please check the input data.")

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        model_score = r2_score(y_test, y_pred) * 100

        with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "wb") as f:
            pickle.dump({"model": model, "features": list(X.columns)}, f)

        update_model_status(model_name, "READY", model_score)
        print(f"Model training complete for {model_name}")
    except Exception as e:
        update_model_status(model_name, "FAILED")
        print(f"Model training failed for {model_name}: {str(e)}")

def start_training(file, model_name, feature):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Invalid CSV file: {str(e)}"}), 400

    save_model_metadata(model_name, feature)

    thread = threading.Thread(target=train_model_thread, args=(df, model_name))
    thread.start()

    return jsonify({"message": "Model training started", "model_name": model_name}), 202
