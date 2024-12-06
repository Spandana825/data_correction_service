import pandas as pd

def load_data(file):
    
    df = pd.read_csv(file)
    #handling missing values is done in analyser. (not sure about righjt way of handling)
    
    return df
