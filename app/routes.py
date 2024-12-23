from flask import Blueprint, request, jsonify
from app.services.train_service import start_training
from app.services.model_service import get_model_status, predict_value, list_models

test_server_bp = Blueprint("test_server", __name__)

train_bp = Blueprint("train", __name__)

status_bp = Blueprint("status", __name__)

list_models_bp = Blueprint("list_models", __name__)

predict_bp = Blueprint("predict", __name__)

@test_server_bp.route("/test", methods=["GET"])
def test_server():
    return "Hi"

@train_bp.route("/train", methods=["POST"])
def train_model():
    if "file" not in request.files or "model_name" not in request.form or "feature" not in request.form:
        return jsonify({"error": "CSV file, feature, and model_name are required"}), 400

    file = request.files["file"]
    model_name = request.form["model_name"]
    feature = request.form["feature"]

    return start_training(file, model_name, feature)

@status_bp.route("/<model_name>/status", methods=["GET"])
def model_status(model_name):
    return get_model_status(model_name)

@list_models_bp.route("/models", methods=["GET"])
def list_user_models():
    return list_models()

@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "model_name" not in data or "payload" not in data:
        return jsonify({"error": "Model name and payload are required"}), 400

    model_name = data["model_name"]
    payload = data["payload"]

    return predict_value(model_name, payload)
