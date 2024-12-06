from flask import request, jsonify
from app import app
from app.services.loader import load_data
from app.services.analyser import train_model,models
from app.services.model import predict_missing_values, get_model

@app.route('/test')
def test():
    return 'HI'
@app.route('/api/models/train', methods=['POST'])
def train_model_api():
    file = request.files['file']
    target_column = request.form['target_column']

    df = load_data(file)
    if target_column not in df.columns:
        return jsonify({"error": f"Target column '{target_column}' not found in data"}), 400
    model_id = train_model(df, target_column)

    return jsonify({"message": "Model training completed", "model_id": model_id}), 200

@app.route('/api/models/status/<model_id>', methods=['GET'])
def model_status(model_id):
    model = get_model(model_id)
    status = "ready" if model else "not found"
    return jsonify({"model_id": model_id, "status": status}), 200

@app.route('/api/models', methods=['GET'])
def list_models():
    return jsonify({"models": list(models.keys())}), 200

@app.route('/api/correction', methods=['POST'])
def data_correction():
    data = request.get_json()
    model_id = data['model_id']
    input_data = data['data']

    try:
        corrected_df = predict_missing_values(model_id, input_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    
    return jsonify({"corrected_data": corrected_df.to_dict(orient='records')}), 200
