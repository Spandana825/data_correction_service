from app.services.analyser import models
import pandas as pd

def get_model(model_id):
    #get model based on the model_id
    return models.get(model_id)

def predict_missing_values(model_id, input_data):
    model = get_model(model_id)
    if not model:
        raise ValueError("Model not found")

    df = pd.DataFrame(input_data)

    # Predict missing values (assuming the missing column is the first one in the DataFrame)
    predictions = model.predict(df.dropna(axis=1))  # Dropping missing columns for prediction

    # Fill missing values with predictions
    missing_column = df.columns[df.isnull().any()].tolist()[0]
    df[missing_column].fillna(predictions, inplace=True)

    return df
