import uuid
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

#store trained models
models = {}

def train_model(df, target_column):
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X) 
    # Initialize  train the model
    model = LinearRegression()
    model.fit(X, y)

    #unique model ID using UUID
    model_id = str(uuid.uuid4())
    models[model_id] = model  

    return model_id
