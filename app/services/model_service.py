import os
import pickle
from flask import jsonify
from models.model_metadata import ModelMetadata, ModelStatus
from sqlalchemy.orm import sessionmaker
from config.database import engine

Session = sessionmaker(bind=engine)

MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

session = Session()

def save_model_metadata(model_name, feature):
    model_metadata = ModelMetadata(
        model_name=model_name,
        feature=feature,
        status=ModelStatus.IN_PROGRESS
    )
    session.add(model_metadata)
    session.commit()
    return model_metadata.model_name

def update_model_status(model_name, status, model_score=None):
    model_metadata = session.query(ModelMetadata).filter_by(model_name=model_name).first()
    if model_metadata:
        model_metadata.status = ModelStatus(status)
        if model_score is not None:
            model_metadata.model_score = model_score
        session.commit()

def get_model_status(model_name):
    model_metadata = session.query(ModelMetadata).filter_by(model_name=model_name).first()
    if not model_metadata:
        return jsonify({"error": "Model not found"}), 404

    return jsonify({"model_name": model_name, "status": model_metadata.status.value}), 200

def list_models():
    models = session.query(ModelMetadata).all()
    return jsonify({
        "models": [
            {
                "model_name": model.model_name,
                "status": model.status.value,
                "model_score": f"{model.model_score:.2f}%",
                "feature": model.feature
            } for model in models
        ]
    }), 200

def predict_value(model_name, payload):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    trained_features = model_data["features"]

    payload_keys = list(payload.keys())
    if not set(trained_features).issubset(set(payload_keys)):
        return jsonify({"error": "Payload contains keys that do not match training features"}), 400

    target_key = None
    input_features = {}
    for key, value in payload.items():
        if value == "" or value is None:
            if target_key is not None:
                return jsonify({"error": "Payload contains multiple empty keys for prediction"}), 400
            target_key = key
        else:
            input_features[key] = value

    if target_key is None:
        return jsonify({"error": "No target key (empty value) found in the payload for prediction"}), 400

    missing_features = set(trained_features) - set(input_features.keys())
    if missing_features:
        return jsonify({"error": f"Payload is missing required features: {list(missing_features)}"}), 400

    input_values = [input_features[feature] for feature in trained_features]
    try:
        prediction = model.predict([input_values])[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    payload[target_key] = prediction

    return jsonify({
        "model_name": model_name,
        "prediction": payload
    }), 200

